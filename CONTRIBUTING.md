# Contributing to Warply

Thanks for your interest in Warply. This project is early-stage; the public API is taking shape, but lifecycle and deployment wiring are still in progress.

## Getting started

```bash
git clone https://github.com/afifi-yusuf/warply.git
cd warply
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the install:

```bash
python -c "import warply as wp; print(wp.__version__)"
```

## What to work on

Current focus (in order):

1. ~~SDK spec objects (`DisaggEngine`, `Pool`) + validation~~
2. **Compiler** — translate spec → provisioning requests, SGLang disagg flags, routing config
3. **Provider plugin** — wrap SkyPilot or Ray for one target cloud
4. **Engine adapter** — SGLang prefill/decode rendering + NIXL KV transfer
5. **Router + client** — basic P/D routing and OpenAI-compatible `client()`
6. **Walking skeleton** — `up() → generate() → scale() → down()` end-to-end

Check open issues and milestones before starting large work. If you plan a significant API or architecture change, open an issue first so we can align on scope.

## Code guidelines

- **Python-first.** The SDK is the product surface; keep user-facing APIs simple and composable.
- **Minimal diffs.** Match existing style and conventions in the files you touch.
- **Validation at the SDK layer.** Fail fast on bad specs before provisioning or deployment.
- **Plugins over monoliths.** Provider, engine, and KV-transfer backends should stay behind small interfaces.
- **No k8s required from users.** YAML/CRD export is an escape hatch, not the primary interface.

## Development

```bash
# Lint
ruff check warply

# Format (if you use ruff format locally)
ruff format warply
```

Run tests when present:

```bash
pytest
```

## Pull requests

1. Fork the repo and create a feature branch from `main`.
2. Keep PRs focused — one logical change per PR when possible.
3. Update `README.md` if you change user-visible behavior or the public API.
4. Describe **what** changed and **why** in the PR body.
5. Link related issues when applicable.

## API stability

Warply is pre-1.0. The `DisaggEngine` / `Pool` surface is intentional, but method semantics may evolve until the walking skeleton lands. Breaking changes should be noted in the PR and release notes.

## Questions

Open a [GitHub issue](https://github.com/afifi-yusuf/warply/issues) for bugs, feature ideas, or design discussion.

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](./LICENSE).
