# Architecture Notes

Warply currently exposes a small SDK surface while the deployment model is still being validated.
This page records vocabulary and architecture decisions that may become public API later.

## Cloud, Runtime, Backend, Endpoint

Warply should keep these concepts separate:

| Concept | Meaning | Examples |
| --- | --- | --- |
| Cloud | Where the GPUs live. | `local`, `lambda`, `aws`, `coreweave`, `modal`, `kubernetes` |
| Runtime | How the serving system is created and managed. | `local_mock`, `raw_skypilot_job`, `skypilot_endpoints`, `modal_endpoint`, `dynamo` |
| Backend | The inference engine that runs tokens. | `sglang`, `vllm`, `tensorrt_llm` |
| Endpoint | The URL clients call after deployment. | `http://.../v1/chat/completions` |

Today, Warply mostly blends cloud and runtime because the implemented cloud path is a raw
SkyPilot task launcher. That is acceptable for v0, but it will become limiting once Warply can
target managed endpoint systems or additional orchestration substrates.

## Why A Runtime Still Exists For Endpoints

An endpoint is the user-facing surface, not the whole system. Something still has to:

- choose clusters and GPUs
- place replicas
- start engine workers and routers
- manage model weights and images
- run health checks
- autoscale
- roll out new versions
- route traffic
- collect logs and metrics
- recover from failures
- tear down resources

That machinery is the runtime or orchestration substrate. For example, a future configuration
could mean:

```python
wp.DisaggEngine(
    model="...",
    backend="sglang",
    cloud="aws",
    runtime="skypilot_endpoints",
)
```

In that shape, AWS provides compute, SkyPilot Endpoints manages serving lifecycle, SGLang runs
tokens, and Warply owns the Python control-plane intent.

## Current Default

The current implementation is equivalent to:

```python
wp.DisaggEngine(
    model="...",
    backend="sglang",
    cloud="lambda",
    # runtime is implicit: raw SkyPilot multi-node task
)
```

Warply should not add a public `runtime=` parameter until there are at least two validated runtime
targets. For now, keep the split internal and documented so future Modal, Dynamo, Kubernetes, or
SkyPilot Endpoints work does not overload `cloud=`.
