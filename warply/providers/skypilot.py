from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

from warply.compiler.plan import ProvisionRequest
from warply.exceptions import ValidationError
from warply.providers.base import Node
from warply.providers.skypilot_task import (
    build_disagg_cluster_task_yaml,
    build_router_task_yaml,
    build_worker_task_yaml,
    cluster_name,
    disagg_cluster_name,
    router_cluster_name,
)

if TYPE_CHECKING:
    from warply.compiler.plan import DeploymentPlan


def _pool_base_port(role: str) -> int:
    return 31000 if role == "prefill" else 32000


class SkyPilotProvider:
    """Provision Warply pools on cloud infrastructure via SkyPilot."""

    def __init__(self, *, plan: DeploymentPlan, session_id: str | None = None) -> None:
        self.plan = plan
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self._router_node: Node | None = None
        self._downed_clusters: set[str] = set()

    def provision(self, request: ProvisionRequest) -> list[Node]:
        if request.replicas != 1:
            raise ValidationError(
                "SkyPilot v0 supports replicas=1 per pool on cloud. "
                "Use replicas=1 for prefill and decode until multireplica routing lands."
            )

        nodes: list[Node] = []
        pool = self.plan.prefill if request.role == "prefill" else self.plan.decode
        for index in range(request.replicas):
            port = _pool_base_port(request.role) + index
            name = cluster_name(session_id=self.session_id, role=request.role, index=index)
            yaml_str = build_worker_task_yaml(
                plan=self.plan,
                pool=pool,
                mode=request.role,
                replica_index=index,
                session_id=self.session_id,
            )
            host = self._launch_task(yaml_str=yaml_str, cluster_name=name)
            nodes.append(
                Node(
                    id=name,
                    role=request.role,
                    host=host,
                    port=port,
                    cluster_name=name,
                    healthy=True,
                )
            )
        return nodes

    def provision_cluster(self) -> tuple[list[Node], list[Node], Node]:
        name = disagg_cluster_name(session_id=self.session_id)
        yaml_str = build_disagg_cluster_task_yaml(plan=self.plan, session_id=self.session_id)
        host = self._launch_task(yaml_str=yaml_str, cluster_name=name)
        prefill = Node(
            id=f"{name}-prefill-0",
            role="prefill",
            host=host,
            port=self.plan.prefill.base_port,
            cluster_name=name,
            healthy=True,
        )
        decode = Node(
            id=f"{name}-decode-0",
            role="decode",
            host=host,
            port=self.plan.decode.base_port,
            cluster_name=name,
            healthy=True,
        )
        router = Node(
            id=f"{name}-router",
            role="router",
            host=host,
            port=self.plan.routing.router_port,
            cluster_name=name,
            healthy=True,
        )
        self._router_node = router
        return [prefill], [decode], router

    def provision_router(self, *, prefill_url: str, decode_url: str) -> Node:
        name = router_cluster_name(session_id=self.session_id)
        yaml_str = build_router_task_yaml(
            plan=self.plan,
            prefill_url=prefill_url,
            decode_url=decode_url,
            session_id=self.session_id,
        )
        host = self._launch_task(yaml_str=yaml_str, cluster_name=name)
        node = Node(
            id=name,
            role="router",
            host=host,
            port=self.plan.routing.router_port,
            cluster_name=name,
            healthy=True,
        )
        self._router_node = node
        return node

    def teardown(self, nodes: list[Node]) -> None:
        for node in nodes:
            if node.role == "router":
                self._router_node = None
            if node.cluster_name in self._downed_clusters:
                continue
            self._downed_clusters.add(node.cluster_name)
            self._down_cluster(node.cluster_name)

    def status(self, nodes: list[Node]) -> list[Node]:
        return nodes

    def _launch_task(self, *, yaml_str: str, cluster_name: str) -> str:
        if os.environ.get("WARPLY_SKYPILOT_DRY_RUN") == "1":
            return f"dryrun.{cluster_name}.example"

        sky = self._import_sky()
        task = sky.Task.from_yaml_str(yaml_str)
        request_id = sky.launch(task, cluster_name=cluster_name)
        _, handle = sky.stream_and_get(request_id)
        return self._handle_head_ip(handle)

    def _down_cluster(self, cluster_name: str) -> None:
        if not cluster_name:
            return
        if os.environ.get("WARPLY_SKYPILOT_DRY_RUN") == "1":
            return

        sky = self._import_sky()
        sky.down(cluster_name)

    @staticmethod
    def _import_sky():
        try:
            import sky
        except ImportError as exc:
            raise ImportError(
                "SkyPilot is required for cloud provisioning. "
                "Install with: pip install warply[cloud]"
            ) from exc
        return sky

    @staticmethod
    def _handle_head_ip(handle) -> str:
        if hasattr(handle, "head_ip") and handle.head_ip:
            return handle.head_ip

        if hasattr(handle, "external_ips"):
            ips = handle.external_ips()
            if ips:
                return ips[0]

        cluster_name = handle.get_cluster_name()
        raise RuntimeError(f"could not resolve external IP for SkyPilot cluster {cluster_name!r}")
