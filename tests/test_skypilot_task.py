from __future__ import annotations

import warply as wp
from warply.providers.skypilot_task import (
    build_router_task_yaml,
    build_worker_task_yaml,
    cluster_name,
)


def make_plan(**overrides):
    kwargs = dict(
        model="meta-llama/Llama-3.1-8B",
        prefill=wp.Pool("1xH100", replicas=1),
        decode=wp.Pool("1xH100", replicas=1),
        cloud="lambda",
    )
    kwargs.update(overrides)
    return wp.DisaggEngine(**kwargs).plan()


def test_worker_task_yaml_contains_sglang_and_lambda():
    plan = make_plan()
    yaml_str = build_worker_task_yaml(
        plan=plan,
        pool=plan.prefill,
        mode="prefill",
        replica_index=0,
        session_id="abc12345",
    )

    assert cluster_name(session_id="abc12345", role="prefill", index=0) in yaml_str
    assert "cloud: lambda" in yaml_str
    assert "H100:1" in yaml_str
    assert "sglang.launch_server" in yaml_str
    assert "--disaggregation-mode" in yaml_str
    assert "prefill" in yaml_str
    assert "--disaggregation-transfer-backend" in yaml_str
    assert "nixl" in yaml_str


def test_router_task_yaml_uses_resolved_urls():
    plan = make_plan()
    yaml_str = build_router_task_yaml(
        plan=plan,
        prefill_url="http://10.0.0.1:31000",
        decode_url="http://10.0.0.2:32000",
        session_id="abc12345",
    )

    assert "sglang_router.launch_router" in yaml_str
    assert "--pd-disaggregation" in yaml_str
    assert "http://10.0.0.1:31000" in yaml_str
    assert "http://10.0.0.2:32000" in yaml_str
