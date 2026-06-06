from __future__ import annotations

from warply.compiler.plan import DeploymentPlan, PoolPlan


class SGLangAdapter:
    """Render SGLang prefill/decode/router process configs."""

    def render_prefill(self, plan: DeploymentPlan) -> dict[str, object]:
        return self._render_worker(plan=plan, pool=plan.prefill, mode="prefill")

    def render_decode(self, plan: DeploymentPlan) -> dict[str, object]:
        return self._render_worker(plan=plan, pool=plan.decode, mode="decode")

    def render_router(self, plan: DeploymentPlan) -> dict[str, object]:
        return {
            "name": "sglang-router",
            "module": "sglang_router.launch_router",
            "argv": [
                "--pd-disaggregation",
                "--host",
                "0.0.0.0",
                "--port",
                str(plan.routing.router_port),
                "--prefill",
                plan.routing.prefill_base_url,
                "--decode",
                plan.routing.decode_base_url,
            ],
            "env": {},
            "port": plan.routing.router_port,
        }

    def health(self, pool: PoolPlan) -> bool:
        return pool.replicas > 0

    def openai_base_url(self, plan: DeploymentPlan) -> str:
        return plan.routing.endpoint

    def _render_worker(
        self,
        *,
        plan: DeploymentPlan,
        pool: PoolPlan,
        mode: str,
    ) -> dict[str, object]:
        return {
            "name": f"sglang-{mode}",
            "module": "sglang.launch_server",
            "argv": [
                "--model-path",
                plan.model,
                "--host",
                "0.0.0.0",
                "--port",
                str(pool.base_port),
                "--tp-size",
                str(pool.gpus_per_replica),
                "--disaggregation-mode",
                mode,
                "--disaggregation-transfer-backend",
                plan.kv_transfer,
            ],
            "env": {},
            "port": pool.base_port,
            "replicas": pool.replicas,
        }
