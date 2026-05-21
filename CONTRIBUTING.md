# Contributing to aimodelground

## Development setup

```powershell
# Clone and install
git clone https://github.com/your-org/aimodelground
cd model-builder

# Install core in editable mode with dev deps
uv sync --dev --project "D:\Projects\model-builder"

# Install plugin packages in editable mode
uv pip install -e packages/model_builder_classical --project "D:\Projects\model-builder"
uv pip install -e packages/model_builder_dl --project "D:\Projects\model-builder"
uv pip install -e packages/model_builder_llm --project "D:\Projects\model-builder"

# Verify
uv run --project "D:\Projects\model-builder" aimodelground --version
```

> **Note (Windows):** `uv sync` removes plugin packages. Re-run the `uv pip install -e` commands after any `uv sync`.

---

## Running tests

```powershell
# Core (connectors, DAG, CLI, web UI, feature store, model update)
uv run --project "D:\Projects\model-builder" pytest tests/ -q

# Plugin packages (run separately to avoid conftest collision)
uv run --project "D:\Projects\model-builder" pytest packages/model_builder_classical/tests/ -q
uv run --project "D:\Projects\model-builder" pytest packages/model_builder_dl/tests/ -q
uv run --project "D:\Projects\model-builder" pytest packages/model_builder_llm/tests/ -q
```

All tests must pass (0 failures) before submitting a PR.

---

## Project layout

```
model_builder/           Core package
  cli/commands/          One file per CLI command
  connectors/            Data source connectors
  core/                  DAG, scheduler, store, gates, runner, events, models
  core_plugins/          Built-in pipeline step plugins
  feature_store/         Feature store (FeatureStore class)
  model_registry.py      Model artifact scanner
  plugins/               Protocol definitions + PluginRegistry
  web/                   FastAPI app + Jinja2 templates

packages/
  model_builder_classical/   Classical ML plugins
  model_builder_dl/          Deep learning plugins
  model_builder_llm/         LLM fine-tuning plugins

tests/                   Core test suite
docs/superpowers/        Design specs and implementation plans
```

---

## Adding a new connector

1. Create `model_builder/connectors/<name>.py`
2. Implement `connect`, `sample`, `profile`, `stream` methods
3. Register in `model_builder/connectors/__init__.py`
4. Register in `PluginRegistry.register_built_ins()` in `model_builder/plugins/registry.py`
5. Write tests in `tests/connectors/test_<name>.py`

Connector protocol (from `model_builder/plugins/protocol.py`):

```python
class Connector(Protocol):
    name: str
    def connect(self, config: dict) -> Connection: ...
    def sample(self, conn: Connection, n: int) -> pd.DataFrame: ...
    def profile(self, conn: Connection) -> DataProfile: ...
    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]: ...
```

---

## Adding a new ML plugin package

1. Create `packages/model_builder_<name>/` with `pyproject.toml`
2. Entry points group: `model_builder.ml_plugins`
3. Entry point format: `"<family>.<algo>" = "module:ClassName"`
4. Implement `MLPlugin` protocol:

```python
class MLPlugin(Protocol):
    name: str
    data_types: list[str]
    requires_gpu: bool
    def detect(self, profile: DataProfile) -> float: ...  # 0.0-1.0 suitability
    def train(self, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact: ...
    def evaluate(self, artifact: Artifact, data: DataBundle) -> EvalReport: ...
    def export(self, artifact: Artifact, fmt: ExportFormat) -> Path: ...
    # Optional:
    def update(self, artifact: Artifact, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact: ...
```

5. Install during dev: `uv pip install -e packages/model_builder_<name> --project "D:\Projects\model-builder"`

---

## Adding a core pipeline plugin

Core plugins are built-in pipeline steps (not external packages).

1. Create `model_builder/core_plugins/<name>_plugin.py`
2. Implement:

```python
class MyPlugin:
    name = "core.my_plugin"

    def run(self, ctx: CoreContext) -> Path:
        # ctx.artifacts_dir — read/write parquets here
        # ctx.run_dir — write receipts, logs here
        # ctx.registry — access other plugins
        # ctx.node_config — dict from pipeline.yaml config:
        ...
        return output_path
```

3. Register in `model_builder/core_plugins/__init__.py`
4. Register in `PluginRegistry.register_built_ins()` in `model_builder/plugins/registry.py`
5. Write tests in `tests/core_plugins/test_<name>.py`

---

## Code style

- Python 3.11+ type hints throughout
- No comments unless the WHY is non-obvious
- No `print()` — use `Rich console` in CLI, `logger` in engine code
- Windows-safe: ASCII only in log messages written to files (cp1252)
- Tests use `pytest-asyncio` with `asyncio_mode = "auto"`
- Async code uses `asyncio.to_thread()` for CPU-bound work
- No global state in connectors or plugins — all config via `connect(config)`

---

## Pull request checklist

- [ ] All core tests pass: `pytest tests/ -q` → 0 failures
- [ ] Relevant plugin tests pass
- [ ] New code has tests
- [ ] No Unicode symbols in log file writes (Windows cp1252 compatibility)
- [ ] `uv sync` still works (no breaking dependency changes)
- [ ] CHANGELOG.md updated under `[Unreleased]`



