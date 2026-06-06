from __future__ import annotations

from warply.compiler.plan import DeploymentPlan


class NixlTransfer:
    """Render NIXL transfer settings consumed by engine adapters."""

    def configure(self, plan: DeploymentPlan) -> dict[str, object]:
        return {
            "backend": "nixl",
            "env": {
                "WARPLY_KV_TRANSFER": "nixl",
                "WARPLY_PREFILL_URL": plan.routing.prefill_base_url,
                "WARPLY_DECODE_URL": plan.routing.decode_base_url,
            },
            "argv": ["--disaggregation-transfer-backend", "nixl"],
        }
