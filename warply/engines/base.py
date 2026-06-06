from __future__ import annotations

from typing import Protocol

from warply.compiler.plan import DeploymentPlan, PoolPlan


class EngineAdapter(Protocol):
    """Renders engine-specific process configs from a deployment plan."""

    def render_prefill(self, plan: DeploymentPlan) -> dict[str, object]: ...

    def render_decode(self, plan: DeploymentPlan) -> dict[str, object]: ...

    def render_router(self, plan: DeploymentPlan) -> dict[str, object]: ...

    def health(self, pool: PoolPlan) -> bool: ...

    def openai_base_url(self, plan: DeploymentPlan) -> str: ...
