from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from warply.runtime.yaml import dump_yaml

if TYPE_CHECKING:
    from warply.compiler.plan import DeploymentPlan, PoolPlan

_CLOUD_ALIASES = {
    "lambda": "lambda",
    "coreweave": "coreweave",
}

_WORKER_SETUP = """\
pip install -U pip
pip install 'sglang[all]'
pip install 'nixl[cu12]' || pip install nixl
"""

_ROUTER_SETUP = """\
pip install -U pip
pip install 'sglang[all]'
"""


def cluster_name(*, session_id: str, role: str, index: int) -> str:
    return f"warply-{session_id}-{role}-{index}"


def router_cluster_name(*, session_id: str) -> str:
    return f"warply-{session_id}-router"


def disagg_cluster_name(*, session_id: str) -> str:
    return f"warply-{session_id}-disagg"


def accelerator_spec(gpu_type: str, count: int) -> str:
    return f"{gpu_type}:{count}"


def _cloud_field(cloud: str) -> str:
    return _CLOUD_ALIASES.get(cloud, cloud)


def argv_to_command(module: str, argv: list[str]) -> str:
    parts = ["python", "-m", module, *argv]
    return " ".join(shlex.quote(part) for part in parts)


def build_worker_task_yaml(
    *,
    plan: DeploymentPlan,
    pool: PoolPlan,
    mode: str,
    replica_index: int,
    session_id: str,
) -> str:
    from warply.engines.sglang import SGLangAdapter

    port = pool.base_port + replica_index
    worker = SGLangAdapter().render_worker(plan=plan, pool=pool, mode=mode, port=port)
    command = argv_to_command(str(worker["module"]), list(worker["argv"]))
    name = cluster_name(session_id=session_id, role=mode, index=replica_index)
    task = {
        "name": name,
        "resources": {
            "cloud": _cloud_field(plan.cloud),
            "accelerators": accelerator_spec(pool.gpu_type, pool.gpus_per_replica),
        },
        "setup": _WORKER_SETUP,
        "run": command,
    }
    return dump_yaml(task)


def build_router_task_yaml(
    *,
    plan: DeploymentPlan,
    prefill_url: str,
    decode_url: str,
    session_id: str,
) -> str:
    from warply.engines.sglang import SGLangAdapter

    routing_plan = replace_routing_urls(plan, prefill_url=prefill_url, decode_url=decode_url)
    router = SGLangAdapter().render_router(routing_plan)
    command = argv_to_command(str(router["module"]), list(router["argv"]))
    name = router_cluster_name(session_id=session_id)
    task = {
        "name": name,
        "resources": {
            "cloud": _cloud_field(plan.cloud),
            "accelerators": accelerator_spec(plan.decode.gpu_type, 1),
        },
        "setup": _ROUTER_SETUP,
        "run": command,
    }
    return dump_yaml(task)


def build_disagg_cluster_task_yaml(*, plan: DeploymentPlan, session_id: str) -> str:
    """Render one multi-node SkyPilot task for v0 prefill/decode disagg."""
    _validate_disagg_cluster_plan(plan)

    from warply.engines.sglang import SGLangAdapter

    adapter = SGLangAdapter()
    prefill = adapter.render_worker(
        plan=plan,
        pool=plan.prefill,
        mode="prefill",
        port=plan.prefill.base_port,
    )
    decode = adapter.render_worker(
        plan=plan,
        pool=plan.decode,
        mode="decode",
        port=plan.decode.base_port,
    )
    router_plan = replace_routing_urls(
        plan,
        prefill_url="WARPLY_PREFILL_URL",
        decode_url="WARPLY_DECODE_URL",
    )
    router = adapter.render_router(router_plan)

    prefill_command = argv_to_command(str(prefill["module"]), list(prefill["argv"]))
    decode_command = argv_to_command(str(decode["module"]), list(decode["argv"]))
    router_command = argv_to_command(str(router["module"]), list(router["argv"]))
    router_command = router_command.replace("WARPLY_PREFILL_URL", '"$PREFILL_URL"')
    router_command = router_command.replace("WARPLY_DECODE_URL", '"$DECODE_URL"')

    run = f"""\
set -euo pipefail
PREFILL_HOST="$(echo "$SKYPILOT_NODE_IPS" | sed -n '1p')"
DECODE_HOST="$(echo "$SKYPILOT_NODE_IPS" | sed -n '2p')"
PREFILL_URL="http://${{PREFILL_HOST}}:{plan.prefill.base_port}"
DECODE_URL="http://${{DECODE_HOST}}:{plan.decode.base_port}"

if [ "${{SKYPILOT_NODE_RANK}}" = "0" ]; then
  {prefill_command} &
  {router_command}
elif [ "${{SKYPILOT_NODE_RANK}}" = "1" ]; then
  {decode_command}
else
  echo "unsupported SKYPILOT_NODE_RANK=${{SKYPILOT_NODE_RANK}}" >&2
  exit 1
fi
"""
    task = {
        "name": disagg_cluster_name(session_id=session_id),
        "resources": {
            "cloud": _cloud_field(plan.cloud),
            "accelerators": accelerator_spec(
                plan.prefill.gpu_type,
                plan.prefill.gpus_per_replica,
            ),
            "num_nodes": 2,
            "network_tier": "best",
        },
        "setup": _WORKER_SETUP,
        "run": run,
    }
    return dump_yaml(task)


def _validate_disagg_cluster_plan(plan: DeploymentPlan) -> None:
    from warply.exceptions import ValidationError

    if plan.prefill.replicas != 1 or plan.decode.replicas != 1:
        raise ValidationError(
            "SkyPilot disagg cluster v0 supports replicas=1 for prefill and decode."
        )
    if plan.prefill.gpu_type != plan.decode.gpu_type:
        raise ValidationError("SkyPilot disagg cluster v0 requires matching GPU types.")
    if plan.prefill.gpus_per_replica != plan.decode.gpus_per_replica:
        raise ValidationError("SkyPilot disagg cluster v0 requires matching GPU counts per node.")


def replace_routing_urls(
    plan: DeploymentPlan,
    *,
    prefill_url: str,
    decode_url: str,
) -> DeploymentPlan:
    from dataclasses import replace

    from warply.compiler.plan import RoutingConfig

    routing = RoutingConfig(
        mode=plan.routing.mode,
        router_port=plan.routing.router_port,
        endpoint=plan.routing.endpoint,
        prefill_base_url=prefill_url,
        decode_base_url=decode_url,
    )
    return replace(plan, routing=routing)
