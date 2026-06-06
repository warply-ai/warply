from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from warply.compiler.plan import ProvisionRequest


@dataclass(frozen=True)
class Node:
    id: str
    role: str
    host: str
    port: int
    cluster_name: str = ""
    healthy: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "role": self.role,
            "host": self.host,
            "port": self.port,
            "cluster_name": self.cluster_name,
            "healthy": self.healthy,
        }


class ProviderPlugin(Protocol):
    """Provider lifecycle interface used by the runtime."""

    def provision(self, request: ProvisionRequest) -> list[Node]: ...

    def teardown(self, nodes: list[Node]) -> None: ...

    def status(self, nodes: list[Node]) -> list[Node]: ...
