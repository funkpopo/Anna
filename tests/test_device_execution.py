from __future__ import annotations

import threading
import time

from anna.runtime.device_execution import DeviceExecutionGate, device_execution_snapshot, process_device_execution


def test_process_device_execution_serializes_owners() -> None:
    order: list[str] = []
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        barrier.wait()
        with process_device_execution(owner=name):
            order.append(f"{name}:enter")
            time.sleep(0.02)
            order.append(f"{name}:exit")

    threads = [
        threading.Thread(target=worker, args=("a",)),
        threading.Thread(target=worker, args=("b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(order) == 4
    # Exclusive sections must nest as enter/exit pairs without interleaving.
    assert order[0].endswith(":enter")
    assert order[1].endswith(":exit")
    assert order[0].split(":")[0] == order[1].split(":")[0]
    snap = device_execution_snapshot()
    assert snap.acquire_count >= 2
    assert snap.holder is None


def test_device_execution_gate_health_fields() -> None:
    gate = DeviceExecutionGate(owner="asr:test")
    with gate.exclusive():
        fields = gate.health_fields()
        assert fields["device_execution_owner"] == "asr:test"
        assert fields["process_device_holder"] == "asr:test"
    fields = gate.health_fields()
    assert fields["process_device_holder"] is None
    assert fields["isolation_mode"] in {"process_serialized", "single_family_preferred"}
