from __future__ import annotations

from warply.compiler.plan import ProvisionRequest
from warply.exceptions import NotReadyError
from warply.providers.base import Node


class SkyPilotProvider:
    """SkyPilot-backed provider stub for cloud integration tests."""

    def provision(self, request: ProvisionRequest) -> list[Node]:
        raise NotReadyError(
            "SkyPilot provisioning is not wired yet. Install warply[cloud] and run the "
            "Lambda integration path once the provider implementation lands."
        )

    def teardown(self, nodes: list[Node]) -> None:
        nodes.clear()

    def status(self, nodes: list[Node]) -> list[Node]:
        return nodes
