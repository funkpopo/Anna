from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


HotPathFindingKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class HotPathFinding:
    path: str
    line: int
    scope: str
    op: str
    source: str

    @property
    def key(self) -> HotPathFindingKey:
        return (self.path, self.scope, self.op)


DEFAULT_HOTPATH_FILES: tuple[str, ...] = (
    "src/anna/runtime/qwen3_5_text_engine.py",
    "src/anna/runtime/scheduler.py",
    "src/anna/runtime/token_staging.py",
    "src/anna/runtime/paged_kv.py",
    "src/anna/runtime/slot_scheduler.py",
    "src/anna/runtime/slot_model_runner.py",
    "src/anna/model/ops.py",
    "src/anna/sampling/sampler.py",
    "src/anna_vllm_xpu/adapter.py",
)


DEFAULT_ALLOWLIST: dict[HotPathFindingKey, int] = {
    ("src/anna/runtime/token_staging.py", "stage_token_ids_to_host", "to_cpu"): 1,
    ("src/anna/runtime/token_staging.py", "stage_token_ids_to_host", "tolist"): 1,
}


class _HotPathVisitor(ast.NodeVisitor):
    def __init__(self, *, path: str, source_lines: Sequence[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.scope: list[str] = []
        self.findings: list[HotPathFinding] = []
        self._cpu_guard_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_If(self, node: ast.If) -> None:
        if self._is_cpu_guard_expr(node.test):
            self._cpu_guard_depth += 1
            try:
                self.visit(node.test)
                for child in node.body:
                    self.visit(child)
            finally:
                self._cpu_guard_depth -= 1
            for child in node.orelse:
                self.visit(child)
            return
        self._record_condition_bool_reductions(node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if self._is_cpu_guard_expr(node.test):
            self._cpu_guard_depth += 1
            try:
                self.visit(node.test)
                for child in node.body:
                    self.visit(child)
            finally:
                self._cpu_guard_depth -= 1
            for child in node.orelse:
                self.visit(child)
            return
        self._record_condition_bool_reductions(node.test)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._record_condition_bool_reductions(node.test)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._record_condition_bool_reductions(node.test)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        op = self._call_op(node)
        if op is not None and self._cpu_guard_depth <= 0:
            self._add_finding(node, op)
        self.generic_visit(node)

    def _scope_name(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def _source_line(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", 0)
        if line <= 0 or line > len(self.source_lines):
            return ""
        return self.source_lines[line - 1].strip()

    def _add_finding(self, node: ast.AST, op: str) -> None:
        self.findings.append(
            HotPathFinding(
                path=self.path,
                line=int(getattr(node, "lineno", 0)),
                scope=self._scope_name(),
                op=op,
                source=self._source_line(node),
            )
        )

    def _record_condition_bool_reductions(self, node: ast.AST) -> None:
        if self._cpu_guard_depth > 0:
            return
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and self._is_torch_bool_reduction(child)
                and not self._is_wrapped_by_bool_call(child, node)
            ):
                self._add_finding(child, "tensor_bool_reduce")

    @staticmethod
    def _is_wrapped_by_bool_call(target: ast.AST, root: ast.AST) -> bool:
        for child in ast.iter_child_nodes(root):
            if child is target:
                return isinstance(root, ast.Call) and isinstance(root.func, ast.Name) and root.func.id == "bool"
            if _HotPathVisitor._is_wrapped_by_bool_call(target, child):
                return True
        return False

    @staticmethod
    def _is_cpu_literal(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and node.value == "cpu":
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "device":
            return bool(node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "cpu")
        return False

    @classmethod
    def _is_cpu_device_type_check(cls, node: ast.AST) -> bool:
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            return False
        comparator = node.comparators[0]
        if not isinstance(node.ops[0], (ast.Eq, ast.Is)) or not cls._is_cpu_literal(comparator):
            return False
        left = node.left
        return isinstance(left, ast.Attribute) and left.attr == "type"

    @classmethod
    def _is_cpu_guard_expr(cls, node: ast.AST) -> bool:
        if cls._is_cpu_device_type_check(node):
            return True
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return bool(node.values) and cls._is_cpu_device_type_check(node.values[0])
        return False

    @staticmethod
    def _is_torch_bool_reduction(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr not in {"all", "any"}:
            return False
        value = node.func.value
        if isinstance(value, ast.Name) and value.id == "torch":
            return True
        return not isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Constant))

    def _call_op(self, node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "bool" and node.args:
            if self._is_torch_bool_reduction(node.args[0]):
                return "tensor_bool_reduce"
            return None
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "torch" and func.attr == "is_nonzero":
                return "torch_is_nonzero"
            if func.attr == "is_nonzero" and not isinstance(
                func.value,
                (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Constant),
            ):
                return "torch_is_nonzero"
        if not isinstance(func, ast.Attribute):
            return None
        if func.attr == "to":
            for arg in node.args:
                if self._is_cpu_literal(arg):
                    return "to_cpu"
            for keyword in node.keywords:
                if keyword.arg == "device" and self._is_cpu_literal(keyword.value):
                    return "to_cpu"
            return None
        if func.attr in {"tolist", "item", "numpy", "cpu"}:
            return func.attr
        return None


def scan_hotpath_file(path: str | Path, *, root: str | Path | None = None) -> list[HotPathFinding]:
    root_path = Path.cwd() if root is None else Path(root)
    source_path = Path(path)
    absolute_path = source_path if source_path.is_absolute() else root_path / source_path
    relative_path = absolute_path.relative_to(root_path).as_posix()
    source = absolute_path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=relative_path)
    visitor = _HotPathVisitor(path=relative_path, source_lines=source_lines)
    visitor.visit(tree)
    return visitor.findings


def scan_hotpath_files(
    paths: Iterable[str | Path] = DEFAULT_HOTPATH_FILES,
    *,
    root: str | Path | None = None,
) -> list[HotPathFinding]:
    findings: list[HotPathFinding] = []
    for path in paths:
        findings.extend(scan_hotpath_file(path, root=root))
    return findings


def summarize_findings(findings: Iterable[HotPathFinding]) -> Counter[HotPathFindingKey]:
    return Counter(finding.key for finding in findings)


def unexpected_findings(
    findings: Iterable[HotPathFinding],
    *,
    allowlist: Mapping[HotPathFindingKey, int] = DEFAULT_ALLOWLIST,
) -> list[HotPathFinding]:
    remaining = Counter(allowlist)
    unexpected: list[HotPathFinding] = []
    for finding in findings:
        if remaining[finding.key] > 0:
            remaining[finding.key] -= 1
            continue
        unexpected.append(finding)
    return unexpected
