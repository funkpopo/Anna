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
    "src/anna/model/ops.py",
    "src/anna/sampling/sampler.py",
)


DEFAULT_ALLOWLIST: dict[HotPathFindingKey, int] = {
    ("src/anna/runtime/qwen3_5_text_engine.py", "AnnaQwen3_5TextEngine._prompt_cache_key", "tolist"): 1,
    ("src/anna/runtime/qwen3_5_text_engine.py", "AnnaQwen3_5TextEngine._prune_trivial_attention_mask", "item"): 1,
    ("src/anna/runtime/qwen3_5_text_engine.py", "AnnaQwen3_5TextEngine._validate_generation_request", "tolist"): 1,
    ("src/anna/runtime/qwen3_5_text_engine.py", "AnnaQwen3_5TextEngine._token_id_from_tensor", "to_cpu"): 1,
    ("src/anna/runtime/token_staging.py", "stage_token_ids_to_host", "to_cpu"): 1,
    ("src/anna/runtime/token_staging.py", "stage_token_ids_to_host", "tolist"): 1,
    ("src/anna/model/ops.py", "Qwen3DynamicCache.set_prompt_token_ids", "tolist"): 1,
    ("src/anna/model/ops.py", "Qwen3DynamicCache._update_visible_layer_cache", "item"): 1,
    ("src/anna/model/ops.py", "Qwen3DynamicCache._update_visible_layer_cache", "tolist"): 1,
    ("src/anna/model/ops.py", "Qwen3DynamicCache._update_turboquant_layer", "tolist"): 1,
    ("src/anna/model/ops.py", "Qwen3DynamicCache.update", "tolist"): 1,
    ("src/anna/model/ops.py", "Qwen3DynamicCache.stack", "tolist"): 1,
    ("src/anna/model/ops.py", "materialized_kv_single_token_decode_attention", "item"): 1,
    ("src/anna/model/ops.py", "Qwen3SparseMoeBlock._execute_offloaded_experts", "numpy"): 1,
    ("src/anna/model/ops.py", "Qwen3SparseMoeBlock.forward._moe_body", "tolist"): 1,
    ("src/anna/model/ops.py", "Qwen3SparseMoeBlock.forward._moe_body", "to_cpu"): 1,
    ("src/anna/model/ops.py", "Qwen3SparseMoeBlock.forward._moe_body", "numpy"): 1,
}


class _HotPathVisitor(ast.NodeVisitor):
    def __init__(self, *, path: str, source_lines: Sequence[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.scope: list[str] = []
        self.findings: list[HotPathFinding] = []

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

    def visit_Call(self, node: ast.Call) -> None:
        op = self._call_op(node)
        if op is not None:
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

    @staticmethod
    def _is_cpu_literal(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and node.value == "cpu":
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "device":
            return bool(node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "cpu")
        return False

    def _call_op(self, node: ast.Call) -> str | None:
        func = node.func
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
        if func.attr in {"tolist", "item", "numpy"}:
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
