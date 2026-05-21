# Changelog

All notable changes to aimodelground are documented here.

## [0.1.0] — 2026-05-21

### Added

**Core infrastructure**
- Async DAG scheduler with asyncio — TASK, GATE, PARALLEL_JOIN node types
- SQLite project store — versioned runs, node state persistence
- DAG loader from `pipeline.yaml` with cycle detection and upstream resolver
- EventBus for async pub/sub between scheduler and web UI (SSE)
- GateManager — approve/skip human-review nodes
- NodeRunner — dispatches to connector/ML/core plugins
- Plugin registry with entry-point auto-discovery

**CLI** (13 commands)
- `init`, `run` (with `--from` for replay), `status`, `approve`, `skip`, `retry`
- `logs`, `runs`, `compare`, `tune`, `export`, `deploy`, `ui`

**Web UI**
- FastAPI + Jinja2 + Bootstrap + HTMX + Plotly
- Pipeline view with live SSE updates, approve/skip buttons
- Data view with file upload, schema, profile stats
- Results view with leaderboard, feature importance, run comparison
- Deploy view with rendered DEPLOY.md

**Data connectors** (9 total)
- `connectors.file` — CSV, JSON, Parquet, Excel, Arrow (DuckDB + glob)
- `connectors.sql` — PostgreSQL, MySQL, SQLite (SQLAlchemy)
- `connectors.rest_poll` — HTTP polling with configurable interval
- `connectors.websocket` — WebSocket stream with JSON parsing
- `connectors.kafka` — Kafka topic consumer (kafka-python, batched)
- `connectors.image` — PNG/JPG/TIFF directory → image_path + metadata
- `connectors.audio` — WAV/MP3/FLAC directory → MFCC features (librosa)
- `connectors.s3` — Amazon S3 via DuckDB httpfs (IAM/keys/MinIO)
- `connectors.gcs` — Google Cloud Storage via DuckDB httpfs

**Core pipeline plugins**
- `core.merge` — concat connector output parquets
- `core.profile` — compute DataProfile from merged data
- `validators.schema` — required columns + null % validation
- `core.automl_ranker` — call `detect()` on all installed ML plugins, rank
- `core.automl_tuner` — Optuna hyperparameter search (CV, configurable trials)
- `core.export` — export trained model (pickle/ONNX/safetensors)
- `core.deploy_advisor` — generate DEPLOY.md with inference code

**ML plugin packages**

`aimodelground-classical`:
- RandomForest (sklearn) — tabular, SHAP importance, ONNX export
- XGBoost — tabular, SHAP importance
- LightGBM — tabular, SHAP importance

`aimodelground-dl`:
- CNNImagePlugin — 3-layer CNN, torchvision ImageFolder
- LSTMTabularPlugin — 2-layer LSTM, gradient-based importance

`aimodelground-llm`:
- LoRATextPlugin — LoRA adapter training on any HF causal/encoder model

**Run versioning**
- Every pipeline execution creates a numbered run
- `run --from <node>` replays from node, reusing upstream outputs
- `compare run_001 run_002` diffs eval metrics



