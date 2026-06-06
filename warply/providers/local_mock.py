from __future__ import annotations

from warply.compiler.plan import ProvisionRequest
from warply.providers.base import Node


class LocalMockProvider:
    """No-GPU provider that simulates healthy local nodes."""

    def provision(self, request: ProvisionRequest) -> list[Node]:
        return [
            Node(
                id=f"local-{request.role}-{index}",
                role=request.role,
                host=f"127.0.0.{index + 1}",
                healthy=True,
            )
            for index in range(request.replicas)
        ]

    def teardown(self, nodes: list[Node]) -> None:
        return None

    def status(self, nodes: list[Node]) -> list[Node]:
        return nodes
