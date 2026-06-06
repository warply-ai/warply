from __future__ import annotations

import warply as wp


engine = wp.DisaggEngine(
    model="meta-llama/Llama-3.1-70B",
    prefill=wp.Pool("4xH100", replicas=2),
    decode=wp.Pool("2xH100", replicas=4),
    cloud="lambda",
)

print(engine.export_yaml())
