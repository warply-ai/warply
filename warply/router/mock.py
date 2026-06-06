from __future__ import annotations

from warply.compiler.plan import DeploymentPlan


class MockRouter:
    """Fixed local router endpoint for no-GPU lifecycle tests."""

    def endpoint(self, plan: DeploymentPlan) -> str:
        return plan.routing.endpoint
