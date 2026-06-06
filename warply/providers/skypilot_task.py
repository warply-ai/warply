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
