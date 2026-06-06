from __future__ import annotations

from typing import Protocol

from warply.compiler.plan import DeploymentPlan


class Router(Protocol):
    """Router interface that exposes an OpenAI-compatible endpoint."""

    def endpoint(self, plan: DeploymentPlan) -> str: ...
