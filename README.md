# Warply

**Warp-level control for disaggregated inference — without the kernel/k8s tax.**

Warply is a Python control plane for disaggregated, self-improving inference. Launch prefill/decode deployments, scale pools independently, and move workloads across clouds — all from `import warply`, not Kubernetes YAML.

> **Status:** Early development. The SDK API is defined; lifecycle and deployment wiring are in progress.

## Why Warply?

Disaggregation mechanisms (prefill/decode separation, KV-cache transfer, routing) are increasingly commoditized — vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, and llm-d all ship them. What's still missing is a **clean, programmable control plane** over those mechanisms.

| Tool | Gap Warply fills |
| --- | --- |
| SkyPilot / SkyServe | Multicloud provisioning, but disagg-unaware |
| NVIDIA Dynamo / llm-d | Strong disagg, but k8s/ops-heavy and not Python-first |
| verl / OpenRLHF | RL orchestration, not a portable serving control plane |
| Managed APIs (Modal, etc.) | Black-box; little control over disagg internals |

Warply sits at the intersection: **SkyPilot's portability + Dynamo's disagg intelligence + researcher-friendly Python**.

## Quick start

```bash
git clone https://github.com/afifi-yusuf/warply.git
cd warply
pip install -e .
```

```python
import warply as wp

engine = wp.DisaggEngine(
    model="meta-llama/Llama-3.1-70B",
    prefill=wp.Pool(gpus="4xH100", replicas=2),
    decode=wp.Pool(gpus="2xH100", replicas=4),
    backend="sglang",
    kv_transfer="nixl",
    cloud="lambda",
)

engine.up()                       # provision + deploy + route (coming soon)
client = engine.client()
resp = client.chat.completions.create(
    model="warply",
    messages=[{"role": "user", "content": "hello"}],
)
engine.down()
```

Context-manager sugar:

```python
with wp.DisaggEngine(...) as engine:
    print(engine.generate("hello"))
```

## MVP scope (v0)

- `DisaggEngine` Python API: declarative spec + `up()` / `scale()` / `down()` / `client()`
- Disaggregated prefill/decode on **one** cloud
- Independent prefill and decode pool scaling
- SGLang engine adapter + NIXL KV transfer
- Basic P/D routing; OpenAI-compatible client from the engine
- Optional `export_yaml()` escape hatch

**Not in v0:** multicloud arbitrage, KV-aware routing, RL/RSI loops, AFD disaggregation. See the [roadmap](#roadmap) below.

## Roadmap

| Phase | Focus |
| --- | --- |
| **0 (now)** | Single-cloud disagg serving via Python SDK |
| **1** | Multicloud providers, KV-aware routing, vLLM/TRT-LLM adapters, RL rollout primitives |
| **2** | Managed control plane, enterprise features, advanced disagg for MoE |

## Design

Warply is a layered control plane:

1. **SDK** — `DisaggEngine` spec + lifecycle methods
2. **Compiler** — spec → provisioning + engine flags + routing
3. **Plugins** — provider, engine, and KV-transfer backends
4. **Router** — prefill pool → decode pool
5. **Observability** — status and health hooks

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and current build priorities.

## Development

```bash
pip install -e ".[dev]"
python -c "import warply as wp; print(wp.__version__)"
```

No local GPU is required for the current walking skeleton. The `cloud="local"` path uses a
mock provider/router so SDK lifecycle tests can exercise `up()`, `generate()`, `scale()`, and
`down()` without starting SGLang.

```python
with wp.DisaggEngine(
    model="meta-llama/Llama-3.1-8B",
    prefill=wp.Pool("1xH100"),
    decode=wp.Pool("1xH100"),
    cloud="local",
) as engine:
    print(engine.generate("hello"))
```

For compiler debugging, use `engine.plan()` for the structured `DeploymentPlan` or
`engine.export_yaml()` for a YAML view of the same artifact.

Current build order:

1. ~~`DisaggEngine` / `Pool` spec + validation~~
2. ~~Compiler: spec → SGLang disagg flags + routing~~
3. ~~Provider plugin skeleton + SGLang adapter + NIXL transfer~~
4. ~~Mock local router + OpenAI-compatible `client()`~~
5. ~~No-GPU walking skeleton: `up() → generate() → scale() → down()`~~
6. SkyPilot Lambda provider implementation
7. GPU-gated SGLang/NIXL integration test on Lambda

Cloud integration tests are intentionally gated. Once the SkyPilot provider is wired, run them
from an environment with credentials and GPUs:

```bash
WARPLY_INTEGRATION=1 pytest tests/test_integration_lambda.py
```

## Contributing

Contributions welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](./LICENSE).
