# model-builder — Project Memory

## What this is

Privacy-first local ML model builder. Async DAG workflow engine with human-in-the-loop gates. Pluggable connectors (file/SQL/stream/cloud) and ML plugin packages (classical/DL/LLM). CLI + FastAPI web UI.

## Key architecture decisions

- **Scheduler exits when gated** — `_is_done()` returns True when no active tasks AND no PENDING nodes have satisfied deps. Prevents infinite loop when downstream nodes are blocked behind a gate.
- **`_resolve_config()` in NodeRunner** — resolves relative file paths against `project_dir` (2 levels above `run_dir`) before passing to connectors. DuckDB uses CWD otherwise.
- **MergePlugin excludes `merged.parquet`** — when collecting parquets to merge, skips the output file itself to avoid self-merge.
- **Core plugins vs ML plugins** — core plugins (`core.*`, `validators.*`) go through `registry.get_core_plugin()` not `get_ml_plugin()`. NodeRunner dispatches based on plugin name prefix.
- **uv sync removes classical/DL/LLM packages** — `uv sync` only keeps `pyproject.toml` deps. Always re-install plugin packages after `uv sync`:
  ```
  uv pip install -e packages/model_builder_classical -e packages/model_builder_dl -e packages/model_builder_llm --project "D:\Projects\model-builder"
  ```
- **Starlette 1.0 TemplateResponse API** — `templates.TemplateResponse(request, "name.html", context_dict)`. Not `(name, {"request": request, ...})`.
- **LLM plugin vocab_size** — test GPT-2 must use `vocab_size=50257` (real tokenizer size), not a smaller value. OOB token errors otherwise.
- **Windows cp1252** — avoid Unicode symbols (✓, ⛔, →) in Rich console output. Use ASCII equivalents.
- **Conftest collision** — running `pytest tests/ packages/*/tests/` together causes `ImportPathMismatchError` on Windows. Run core and plugin test suites separately.

## Project structure

```
model_builder/           Core package (connectors, DAG, scheduler, CLI, web UI)
  cli/commands/          One file per CLI command
  connectors/            All data connectors (file, sql, rest, ws, kafka, image, audio, s3, gcs)
  core/                  DAG, scheduler, store, gates, runner, events, models
  core_plugins/          Built-in pipeline plugins (merge, profile, validate, rank, tune, export, deploy)
  plugins/               Protocol definitions + registry
  web/                   FastAPI app + Jinja2 templates

packages/
  model_builder_classical/   RandomForest, XGBoost, LightGBM (pip install model-builder-classical)
  model_builder_dl/          CNN, LSTM (pip install model-builder-dl)
  model_builder_llm/         LoRA text fine-tuning (pip install model-builder-llm)
```

## Test commands

```
# Core (connectors, DAG, CLI, web)
uv run --project "D:\Projects\model-builder" pytest tests/ -q

# Classical ML
uv run --project "D:\Projects\model-builder" pytest packages/model_builder_classical/tests/ -q

# Deep learning
uv run --project "D:\Projects\model-builder" pytest packages/model_builder_dl/tests/ -q

# LLM
uv run --project "D:\Projects\model-builder" pytest packages/model_builder_llm/tests/ -q
```

## Adding a new connector

1. Create `model_builder/connectors/<name>.py` implementing `connect`, `sample`, `profile`, `stream`
2. Add to `model_builder/connectors/__init__.py`
3. Register in `PluginRegistry.register_built_ins()` with key `connectors.<name>`
4. Write tests in `tests/connectors/test_<name>.py`

## Adding a new ML plugin package

1. Create `packages/model_builder_<name>/` with own `pyproject.toml`
2. Entry points group: `model_builder.ml_plugins`
3. Entry point name format: `<family>.<algorithm>` → registry key `ml.<family>.<algorithm>`
4. Implement `MLPlugin` protocol: `detect`, `train`, `evaluate`, `export`
5. Install: `uv pip install -e packages/model_builder_<name> --project "D:\Projects\model-builder"`

## Adding a new core plugin

1. Create `model_builder/core_plugins/<name>_plugin.py` with class implementing `run(ctx: CoreContext) -> Path`
2. Register in `PluginRegistry.register_built_ins()` with key `core.<name>`
3. Extend `CoreContext` if the plugin needs new context fields

## Node runner dispatch order

```
plugin prefix → dispatcher
connectors.*  → _run_connector (FileConnector, SQLConnector, etc.)
ml.*          → _run_ml_plugin (registry.get_ml_plugin)
core.* / validators.*  → _run_core_plugin (registry.get_core_plugin)
GATE          → AWAITING_HUMAN (no execution)
PARALLEL_JOIN → immediately SUCCEEDED
```

## CLI commands (current)

init, run, status, approve, skip, retry, logs, runs, compare, tune, export, deploy, ui
