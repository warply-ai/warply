from __future__ import annotations

from typing import Protocol

from warply.compiler.plan import DeploymentPlan


class KVTransfer(Protocol):
    """Renders KV-transfer-specific flags and environment."""

    def configure(self, plan: DeploymentPlan) -> dict[str, object]: ...
