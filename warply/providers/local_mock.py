from __future__ import annotations

from typing import TYPE_CHECKING

from warply.compiler.plan import ProvisionRequest
from warply.providers.base import Node

if TYPE_CHECKING:
    from warply.compiler.plan import DeploymentPlan


def _pool_base_port(role: str) -> int:
    return 31000 if role == "prefill" else 32000


class LocalMockProvider:
    """No-GPU provider that simulates healthy local nodes."""

    def __init__(self, *, plan: DeploymentPlan | None = None) -> None:
        self.plan = plan

    def provision(self, request: ProvisionRequest) -> list[Node]:
        base_port = _pool_base_port(request.role)
        return [
            Node(
                id=f"local-{request.role}-{index}",
                role=request.role,
                host="127.0.0.1",
                port=base_port + index,
                cluster_name=f"local-{request.role}-{index}",
                healthy=True,
            )
            for index in range(request.replicas)
        ]

    def provision_router(self, *, prefill_url: str = "", decode_url: str = "") -> Node:
        return Node(
            id="local-router",
            role="router",
            host="127.0.0.1",
            port=8000,
            cluster_name="local-router",
            healthy=True,
        )

    def provision_cluster(self) -> tuple[list[Node], list[Node], Node]:
        if self.plan is None:
            raise RuntimeError(
                "LocalMockProvider requires a deployment plan for provision_cluster()."
            )

        prefill_nodes = self.provision(self.plan.prefill.provision)
        decode_nodes = self.provision(self.plan.decode.provision)
        router = self.provision_router()
        return prefill_nodes, decode_nodes, router

    def teardown(self, nodes: list[Node]) -> None:
        return None

    def status(self, nodes: list[Node]) -> list[Node]:
        return nodes
