from __future__ import annotations

from dataclasses import dataclass, field

from warply.compiler.plan import DeploymentPlan
from warply.providers.base import Node, ProviderPlugin
from warply.providers.local_mock import LocalMockProvider
from warply.router.mock import MockRouter
from warply.runtime.client import MockOpenAIClient
from warply.types import EngineState


@dataclass
class Runtime:
    """State machine that binds a plan to provider/router plugins."""

    plan: DeploymentPlan
    provider: ProviderPlugin = field(default_factory=LocalMockProvider)
    router: MockRouter = field(default_factory=MockRouter)
    prefill_nodes: list[Node] = field(default_factory=list)
    decode_nodes: list[Node] = field(default_factory=list)
    endpoint: str | None = None

    def up(self) -> None:
        self.prefill_nodes = self.provider.provision(self.plan.prefill.provision)
        self.decode_nodes = self.provider.provision(self.plan.decode.provision)
        self.endpoint = self.router.endpoint(self.plan)

    def scale(self, *, prefill: int | None = None, decode: int | None = None) -> None:
        if prefill is not None:
            request = self.plan.prefill.provision
            self.provider.teardown(self.prefill_nodes)
            self.prefill_nodes = self.provider.provision(
                request.__class__(
                    role=request.role,
                    cloud=request.cloud,
                    gpu_type=request.gpu_type,
                    gpus_per_replica=request.gpus_per_replica,
                    replicas=prefill,
                )
            )
        if decode is not None:
            request = self.plan.decode.provision
            self.provider.teardown(self.decode_nodes)
            self.decode_nodes = self.provider.provision(
                request.__class__(
                    role=request.role,
                    cloud=request.cloud,
                    gpu_type=request.gpu_type,
                    gpus_per_replica=request.gpus_per_replica,
                    replicas=decode,
                )
            )

    def down(self) -> None:
        self.provider.teardown(self.prefill_nodes)
        self.provider.teardown(self.decode_nodes)
        self.prefill_nodes = []
        self.decode_nodes = []
        self.endpoint = None

    def client(self) -> MockOpenAIClient:
        return MockOpenAIClient(base_url=self.endpoint or self.plan.routing.endpoint)

    def healthy_prefill(self) -> int:
        return sum(node.healthy for node in self.provider.status(self.prefill_nodes))

    def healthy_decode(self) -> int:
        return sum(node.healthy for node in self.provider.status(self.decode_nodes))


def state_after_down(current: EngineState) -> EngineState:
    stoppable_states = {
        EngineState.PENDING,
        EngineState.READY,
        EngineState.SCALING,
        EngineState.STOPPING,
    }
    if current in stoppable_states:
        return EngineState.STOPPED
    return current
