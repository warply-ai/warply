from __future__ import annotations

import pytest

import warply as wp
from warply.exceptions import ValidationError
from warply.providers.skypilot_task import (
    build_disagg_cluster_task_yaml,
    build_router_task_yaml,
    build_worker_task_yaml,
    cluster_name,
    disagg_cluster_name,
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


def test_disagg_cluster_task_yaml_uses_two_node_best_network():
    plan = make_plan()
    yaml_str = build_disagg_cluster_task_yaml(plan=plan, session_id="abc12345")

    assert disagg_cluster_name(session_id="abc12345") in yaml_str
    assert "cloud: lambda" in yaml_str
    assert "accelerators: 'H100:1'" in yaml_str
    assert "num_nodes: 2" in yaml_str
    assert "network_tier: best" in yaml_str
    assert "SKYPILOT_NODE_RANK" in yaml_str
    assert "SKYPILOT_NODE_IPS" in yaml_str
    assert "sglang.launch_server" in yaml_str
    assert "--disaggregation-mode prefill" in yaml_str
    assert "--disaggregation-mode decode" in yaml_str
    assert "sglang_router.launch_router" in yaml_str
    assert "--disaggregation-transfer-backend nixl" in yaml_str
    assert yaml_str.count("accelerators:") == 1


def test_disagg_cluster_task_rejects_multireplica_cloud_plan():
    plan = make_plan(prefill=wp.Pool("1xH100", replicas=2))

    with pytest.raises(ValidationError, match="replicas=1"):
        build_disagg_cluster_task_yaml(plan=plan, session_id="abc12345")


def test_disagg_cluster_task_rejects_mismatched_gpu_type():
    plan = make_plan(decode=wp.Pool("1xA100", replicas=1))

    with pytest.raises(ValidationError, match="GPU types"):
        build_disagg_cluster_task_yaml(plan=plan, session_id="abc12345")


def test_disagg_cluster_task_rejects_mismatched_gpu_count():
    plan = make_plan(decode=wp.Pool("2xH100", replicas=1))

    with pytest.raises(ValidationError, match="GPU counts"):
        build_disagg_cluster_task_yaml(plan=plan, session_id="abc12345")
