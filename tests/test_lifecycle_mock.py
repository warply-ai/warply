from __future__ import annotations

import pytest

import warply as wp
from warply.exceptions import NotReadyError
from warply.providers.base import Node
from warply.runtime.lifecycle import Runtime
from warply.types import EngineState


def make_engine(**overrides) -> wp.DisaggEngine:
    kwargs = dict(
        model="meta-llama/Llama-3.1-8B",
        prefill=wp.Pool("1xH100", replicas=1),
        decode=wp.Pool("1xH100", replicas=1),
        cloud="local",
    )
    kwargs.update(overrides)
    return wp.DisaggEngine(**kwargs)


def test_local_mock_lifecycle_walks_without_gpu():
    engine = make_engine()

    engine.up()
    status = engine.status()

    assert status.state is EngineState.READY
    assert status.endpoint == "http://127.0.0.1:8000"
    assert status.prefill.healthy_replicas == 1
    assert status.decode.healthy_replicas == 1

    assert engine.generate("hi").endswith("hi")

    engine.scale(decode=3)
    scaled = engine.status()
    assert scaled.state is EngineState.READY
    assert scaled.decode.replicas == 3
    assert scaled.decode.healthy_replicas == 3

    engine.down()
    assert engine.status().state is EngineState.STOPPED
    assert engine.status().endpoint is None


def test_context_manager_uses_mock_lifecycle():
    with make_engine() as engine:
        assert engine.status().state is EngineState.READY
        assert engine.client().base_url == "http://127.0.0.1:8000"

    assert engine.status().state is EngineState.STOPPED


def test_up_rejects_already_running_engine():
    engine = make_engine()
    engine.up()

    with pytest.raises(NotReadyError):
        engine.up()

    assert engine.status().state is EngineState.READY
    assert engine.status().prefill.healthy_replicas == 1


def test_non_local_up_uses_skypilot_dry_run(monkeypatch):
    monkeypatch.setenv("WARPLY_SKYPILOT_DRY_RUN", "1")
    engine = make_engine(
        cloud="lambda",
        prefill=wp.Pool("1xH100", replicas=1),
        decode=wp.Pool("1xH100", replicas=1),
    )

    engine.up()
    assert engine.status().state is EngineState.READY
    engine.down()


def test_runtime_plan_tracks_scaled_replicas():
    engine = make_engine(decode=wp.Pool("1xH100", replicas=2))

    engine.up()
    engine.scale(decode=3)

    assert engine._runtime is not None
    assert engine._runtime.plan.decode.replicas == engine.plan().decode.replicas
    assert engine._runtime.plan.decode.provision.replicas == 3


class _FailSecondProvisionProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.torn_down: list[str] = []

    def provision(self, request) -> list[Node]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("provision failed")
        return [
            Node(
                id=f"{request.role}-{self.calls}",
                role=request.role,
                host="127.0.0.1",
                port=31000,
            )
        ]

    def teardown(self, nodes: list[Node]) -> None:
        self.torn_down.extend(node.id for node in nodes)

    def status(self, nodes: list[Node]) -> list[Node]:
        return nodes


def test_up_failure_cleans_runtime_and_stops_engine(monkeypatch):
    provider = _FailSecondProvisionProvider()

    def make_runtime(plan):
        return Runtime(plan, provider=provider)

    monkeypatch.setattr("warply.engine.create_runtime", make_runtime)
    engine = make_engine()

    with pytest.raises(RuntimeError, match="provision failed"):
        engine.up()

    assert engine.status().state is EngineState.STOPPED
    assert engine._runtime is None
    assert provider.torn_down == ["prefill-1"]


class _FailDecodeScaleProvider:
    def provision(self, request) -> list[Node]:
        if request.role == "decode" and request.replicas == 3:
            raise RuntimeError("scale failed")
        return [
            Node(
                id=f"{request.role}-{index}",
                role=request.role,
                host="127.0.0.1",
                port=32000 + index,
            )
            for index in range(request.replicas)
        ]

    def teardown(self, nodes: list[Node]) -> None:
        return None

    def status(self, nodes: list[Node]) -> list[Node]:
        return nodes


def test_scale_failure_preserves_engine_and_runtime_state():
    engine = make_engine(decode=wp.Pool("1xH100", replicas=2))
    engine.up()
    assert engine._runtime is not None
    engine._runtime.provider = _FailDecodeScaleProvider()

    with pytest.raises(RuntimeError, match="scale failed"):
        engine.scale(decode=3)

    assert engine.status().state is EngineState.READY
    assert engine.status().decode.replicas == 2
    assert engine.status().decode.healthy_replicas == 2
    assert engine.plan().decode.replicas == 2
    assert engine._runtime.plan.decode.replicas == 2
