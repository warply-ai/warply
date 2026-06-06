from __future__ import annotations

from warply.compiler.plan import DeploymentPlan
from warply.providers.local_mock import LocalMockProvider
from warply.providers.skypilot import SkyPilotProvider
from warply.runtime.lifecycle import Runtime


def create_runtime(plan: DeploymentPlan) -> Runtime:
    """Create a runtime wired for the target cloud in the deployment plan."""
    if plan.cloud == "local":
        return Runtime(plan, provider=LocalMockProvider(plan=plan))

    return Runtime(plan, provider=SkyPilotProvider(plan=plan))
