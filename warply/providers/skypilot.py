from __future__ import annotations

from warply.compiler.plan import ProvisionRequest
from warply.providers.base import Node


class SkyPilotProvider:
    """SkyPilot-backed provider stub for cloud integration tests."""

    def provision(self, request: ProvisionRequest) -> list[Node]:
        raise NotImplementedError(
            "SkyPilot provisioning is not wired yet. Install warply[cloud] and run the "
            "Lambda integration path once the provider implementation lands."
        )

    def teardown(self, nodes: list[Node]) -> None:
        return None

    def status(self, nodes: list[Node]) -> list[Node]:
        return nodes
