"""Supported plugin identifiers for the MVP SDK surface."""

SUPPORTED_BACKENDS = frozenset({"sglang"})
SUPPORTED_KV_TRANSFERS = frozenset({"nixl"})
SUPPORTED_CLOUDS = frozenset({"lambda", "coreweave", "local"})

GPU_SPEC_PATTERN = r"^\d+x[A-Za-z0-9][A-Za-z0-9._-]*$"
