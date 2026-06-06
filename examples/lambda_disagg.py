from __future__ import annotations

import warply as wp


engine = wp.DisaggEngine(
    model="meta-llama/Llama-3.1-8B",
    prefill=wp.Pool("1xH100", replicas=1),
    decode=wp.Pool("1xH100", replicas=1),
    cloud="lambda",
)

print(engine.export_yaml())
