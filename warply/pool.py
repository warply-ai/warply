from __future__ import annotations

import re
from dataclasses import dataclass

from warply._constants import GPU_SPEC_PATTERN
from warply.exceptions import ValidationError


@dataclass(frozen=True)
class Pool:
    """One side of a disaggregated deployment (prefill or decode)."""

    gpus: str
    replicas: int = 1

    def __post_init__(self) -> None:
        if not re.fullmatch(GPU_SPEC_PATTERN, self.gpus):
            raise ValidationError(
                f"Invalid gpus spec {self.gpus!r}; expected format like '4xH100'."
            )
        if self.replicas < 1:
            raise ValidationError(f"replicas must be >= 1, got {self.replicas}.")
