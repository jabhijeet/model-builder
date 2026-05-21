# Changelog

All notable changes to aimodelground are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

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
