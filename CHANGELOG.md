# Changelog

All notable changes to aimodelground are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.3.0] — 2026-05-21

### Added

**Guided 6-step wizard UI** — replaces the flat 4-tab Bootstrap layout with a contextual step-by-step flow:

| Step | Page | What changed |
|------|------|-------------|
| 1 | **Upload** | Drag-and-drop zone, file pills showing name + size + "ready" badge, auto-advance hint |
| 2 | **Configure** | Split-pane: smart form (file selector, target column auto-detected, algorithm chips) + live YAML textarea that syncs as you change form fields. Validate + Save buttons. |
| 3 | **Run** | **Run Pipeline button** — triggers pipeline directly from UI (no terminal needed). "From node" dropdown for partial re-runs. Gate approval cards, progress bar, live node list. |
| 4 | **Results** | Metric summary cards (big numbers), SHAP feature importance bars, Plotly chart, run compare delta table |
| 5 | **Deploy** | Query Model link, export info, algorithm rankings, copy buttons |
| 6 | **Query** | Predict tab (ML model inference from feature inputs) + Explain tab (SHAP + metrics + pre-written insights, no LLM) |

**Wizard stepper** — sticky top bar shows all 6 steps. Completed steps are clickable (green ✓), current step is highlighted (blue), locked steps are grey. Max unlocked step is computed from project state (files → yaml → run → deploy → model export).

**Deep Space theme** — custom CSS variables, no Bootstrap dependency:
- Dark default: `#0a0e1a` background, `#4f8ef7` accent blue, `#00d4a0` success green
- Light mode toggle: `data-theme="light"` on `<html>`, preference stored in `localStorage`
- Inter font (Google Fonts CDN), weights 400–800
- All CSS via `<style>` block in `base.html` — no external CSS framework

**UI-triggered pipeline runs** (`POST /api/run`):
- Calls FastAPI → `asyncio.create_task` → runs scheduler in background
- EventBus subscriber wires directly to SSE broadcaster — live node updates flow to browser during run
- Supports `?from_node=<id>` for partial replay
- No terminal required for basic use

**New API endpoints**:
- `GET /api/file-info/{filename}` — pandas column detection, dtype inference, target heuristic (looks for columns named `target`, `label`, `y`, `class`, `species` etc.)
- `GET /api/yaml` — read current `pipeline.yaml`
- `POST /api/yaml` — write and validate `pipeline.yaml`
- `POST /api/yaml/validate` — dry-run YAML parse + `nodes` key check, no write
- `POST /api/predict` — load exported model (joblib), run `.predict()` + `.predict_proba()`, return `{prediction, confidence, top_feature, top_feature_value}`
- `GET /api/explain` — aggregate `eval_report.json` + `profile.json` into `{metrics, feature_importance, profile, insights}`

**New routes**: `/upload`, `/configure`, `/query`

**Pre-written insights** (Explain tab, no LLM):
- Accuracy < 0.7 → "consider more data or hyperparameter tuning"
- Top feature dominance > 0.8 → "model may overfit to this feature"
- Column null rate > 10% → "clean source data"

### Changed

- `base.html` fully rewritten — Bootstrap 5 removed, custom CSS variables, Inter font, wizard stepper
- `pipeline.html` renamed to `run.html` with new Run button, gate cards, progress bar
- `data.html` removed — `/data` now redirects (302) to `/upload`
- `pipeline_nodes.html` restyled — new badge/card CSS, Approve/Skip/Retry buttons use new classes
- `results.html` restyled — metric cards, SHAP horizontal bars, Plotly transparent background
- `deploy.html` restyled — Quick Actions sidebar, Query Model link, copy buttons
- All page routes now receive `current_step` and `max_step` for wizard stepper rendering
- `_step_context()` helper computes `max_step` from live project state on every request

### Fixed

- `predict.js` — XSS: all server-sourced values (column names, metric keys, insight strings) are HTML-escaped via `escHtml()` before `innerHTML` insertion
- `predict.js` — network errors during prediction no longer permanently disable the Predict button (try/catch/finally)
- `yaml-editor.js` — `saveYaml()` and `validateYaml()` now null-check DOM elements before access

---

## [0.2.0] — 2026-05-21

### Added

**Document connector**
- `connectors.document` — extracts text from PDF, DOCX, DOC, TXT, MD, RST files
- Page-by-page extraction for PDFs (one row per page with filename, page, total_pages, text, char_count)
- `label_from_dir: true` — uses parent directory name as label column (for classification tasks)
- Glob pattern support: `["data/raw/**/*.pdf", "data/raw/**/*.docx"]`
- Returns `data_types: ["text"]` so AutoML ranker routes to `aimodelground-llm` automatically
- Dependencies: `pypdf>=4.0`, `python-docx>=1.1`

**Feature store**
- `FeatureStore` class — SQLite-backed versioned feature set registry (`.modelbuilder/feature_store.db`)
- `connectors.feature_store` — load saved feature sets into a pipeline by name + version
- `core.feature_store_save` pipeline node — save pipeline artifacts as named, tagged feature sets
- CLI: `aimodelground features list|info|versions|delete`
- Versioned: each `save()` auto-increments version; load latest or pin to specific version

**Model update**
- `scan_models()` / `ModelRegistry` — scans all run directories for trained model artifacts
- `RandomForestPlugin.update()` — warm-start (adds trees without retraining from scratch)
- `XGBoostPlugin.update()` — incremental training via `xgb_model=existing_booster`
- `LightGBMPlugin.update()` — incremental training via `init_model=existing_model`
- `core.model_update` pipeline node — updates existing model artifact with new data
- CLI: `aimodelground models list|info|update`

**Hyperparameter tuning**
- `core.automl_tuner` — Optuna-based cross-validated hyperparameter search
- Search spaces for RandomForest, XGBoost, LightGBM
- CLI: `aimodelground tune --trials N --cv K --target col`
- Results saved as `tuning_results.json` in the run artifacts directory

**Cloud storage connectors**
- `connectors.s3` — Amazon S3 via DuckDB httpfs (IAM roles, access keys, MinIO/LocalStack via `endpoint_url`)
- `connectors.gcs` — Google Cloud Storage via DuckDB httpfs (HMAC keys, service account JSON)
- Both support glob paths, auto-detect format from file extension, zero extra dependencies (DuckDB built-in)

**Image + Audio connectors**
- `connectors.image` — reads PNG/JPG/TIFF from directory, produces DataFrame with image_path, label, width, height, mode
- `connectors.audio` — reads WAV/MP3/FLAC, extracts MFCC features via librosa (soundfile backend for WAV/FLAC to avoid deprecated audioop)

**Kafka connector**
- `connectors.kafka` — Kafka topic consumer (kafka-python), configurable batch_size and max_messages

**Remaining CLI commands**
- `aimodelground logs <node>` — tail last N lines of node execution log
- `aimodelground export --format <fmt>` — re-export trained model (pickle/onnx/safetensors)
- `aimodelground deploy [--regenerate]` — print or regenerate DEPLOY.md
- `aimodelground tune` — run Optuna hyperparameter search
- `aimodelground features` — feature store subcommands
- `aimodelground models` — model registry subcommands
- `aimodelground --version` / `-V` — print version and exit

**Web UI improvements**
- Version number shown in navbar (`v0.2.0`)
- **Pipeline page**: contextual banners (no run / gate waiting / running / complete), node status legend, "What to do next" card
- **Data page**: step-by-step instructions, null warning highlights (>10%), "Next steps" card
- **Results page**: metric direction hints (higher/lower is better), run comparison links, feature importance caption, "What to do next" panel
- **Deploy page**: Copy button for full guide, Copy path for model file, quick actions sidebar

**Deep learning plugins** (`aimodelground-dl`)
- `CNNImagePlugin` — 3-layer CNN for image classification (torchvision ImageFolder)
- `LSTMTabularPlugin` — 2-layer LSTM for sequential/tabular data, gradient-based feature importance

**LLM fine-tuning plugin** (`aimodelground-llm`)
- `LoRATextPlugin` — LoRA fine-tuning on GPT-2, Llama, Mistral, Phi, BERT via HuggingFace PEFT
- Auto-detects text column by heuristic (longest string column)
- Saves adapter weights as safetensors (compatible with Ollama, vLLM)
- Pads token ID set correctly on model reload

### Changed

- Package renamed from `model-builder` to `aimodelground` (PyPI name conflict resolved)
- License changed from MIT to **Apache 2.0**; `NOTICE` file added
- `datetime.datetime.utcnow()` replaced with `datetime.datetime.now(datetime.UTC)` throughout (Python 3.12 deprecation fix)
- Audio connector now uses `soundfile` backend for WAV/FLAC (avoids deprecated `audioop` module)
- `pyproject.toml`: dynamic version from `model_builder/__init__.py`, full PyPI classifiers, sdist include list, `[tool.hatch.version]` config

### Fixed

- Scheduler `_is_done()` now exits when no pending nodes are dispatchable (previously looped forever when downstream nodes were blocked behind a gate)
- NodeRunner `_resolve_config()` resolves relative file paths against project dir (DuckDB was resolving against CWD)
- MergePlugin excludes `merged.parquet` from its own input list (prevented self-merge)
- Starlette 1.0 `TemplateResponse` API updated to `(request, name, context)` signature
- Windows cp1252 encoding — Unicode symbols removed from log file writes and CLI output

---

## [0.1.0] — 2026-05-21

### Added

**Core infrastructure**
- Async DAG scheduler — TASK, GATE, PARALLEL_JOIN node types
- SQLite project store — versioned runs, node state persistence
- DAG loader from `pipeline.yaml` with cycle detection and upstream resolver
- EventBus for async pub/sub (SSE)
- GateManager — approve/skip human-review nodes
- NodeRunner — dispatches to connector/ML/core plugins
- Plugin registry with entry-point auto-discovery

**CLI** — `init`, `run` (`--from` replay), `status`, `approve`, `skip`, `retry`, `runs`, `compare`, `ui`

**Web UI** — FastAPI + Jinja2 + Bootstrap + HTMX + Plotly; Pipeline / Data / Results / Deploy views

**Data connectors** — `file`, `sql`, `rest_poll`, `websocket`

**Core plugins** — `merge`, `profile`, `validators.schema`, `automl_ranker`, `export`, `deploy_advisor`

**ML plugins** — `aimodelground-classical` (RandomForest, XGBoost, LightGBM with SHAP + ONNX)

**Run versioning** — numbered runs, `run --from`, `compare`
