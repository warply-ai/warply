from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EngineState(str, Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    READY = "ready"
    SCALING = "scaling"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass(frozen=True)
class PoolStatus:
    gpus: str
    replicas: int
    healthy_replicas: int = 0


@dataclass(frozen=True)
class DeploymentStatus:
    state: EngineState
    model: str
    backend: str
    kv_transfer: str
    cloud: str
    prefill: PoolStatus
    decode: PoolStatus
    endpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "model": self.model,
            "backend": self.backend,
            "kv_transfer": self.kv_transfer,
            "cloud": self.cloud,
            "prefill": {
                "gpus": self.prefill.gpus,
                "replicas": self.prefill.replicas,
                "healthy_replicas": self.prefill.healthy_replicas,
            },
            "decode": {
                "gpus": self.decode.gpus,
                "replicas": self.decode.replicas,
                "healthy_replicas": self.decode.healthy_replicas,
            },
            "endpoint": self.endpoint,
        }
