from __future__ import annotations

import pytest

import warply as wp
from warply.exceptions import NotReadyError
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


def test_non_local_up_is_not_implemented_yet():
    engine = make_engine(cloud="lambda")

    with pytest.raises(NotImplementedError):
        engine.up()
