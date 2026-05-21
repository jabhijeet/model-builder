# aimodelground

[![PyPI version](https://img.shields.io/pypi/v/aimodelground.svg)](https://pypi.org/project/aimodelground/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache-yellow.svg)](LICENSE)

**Privacy-first, locally-installed ML model builder.**

Upload data from any source, let the app guide you step-by-step through training, and get a deployable model — entirely on your machine. No cloud, no telemetry, no data leaving your system.

---

## Installation

```bash
pip install aimodelground
```

Then install ML plugins based on your data type:

| Plugin | Install when you have | Examples |
|--------|-----------------------|---------|
| `aimodelground-classical` | **Tabular / structured data** — spreadsheets, SQL exports, CSVs with numeric/categorical columns. Best default choice. Fast, runs on any machine, no GPU needed. | Customer churn, fraud detection, price prediction, sales forecasting |
| `aimodelground-dl` | **Images or sequences** — folders of photos/scans, or time-series data where row order matters. Needs more RAM. GPU optional but speeds up training significantly. | Image classification, defect detection, sensor anomaly detection, log sequence analysis |
| `aimodelground-llm` | **Text data** — product reviews, support tickets, emails, documents. Fine-tunes an existing language model (GPT-2, Llama, Mistral) on your labels. GPU strongly recommended (8GB+ VRAM for Llama/Mistral; CPU-only works for GPT-2). | Sentiment analysis, topic classification, intent detection, document routing |

```bash
# Tabular data (CSV, SQL, Excel) — install this first, covers most use cases
pip install aimodelground-classical

# Image or sequential data — requires PyTorch (~2GB download)
pip install aimodelground-dl

# Text classification with LLM fine-tuning — requires PyTorch + HuggingFace (~500MB + model weights)
pip install aimodelground-llm

# Or install everything at once
pip install aimodelground-classical aimodelground-dl aimodelground-llm
```

> **Not sure?** Start with `aimodelground-classical`. The AutoML ranker will tell you which algorithms suit your data after profiling.

**Requires Python 3.11+**

---

## Quick start

```bash
aimodelground init my-model      # create project
cp data.csv my-model/data/raw/  # add your data
cd my-model
aimodelground run               # start pipeline
aimodelground approve review_data  # approve a gate
aimodelground run               # continue
aimodelground ui                # open web interface
aimodelground deploy            # view deployment guide
```

---

## How it works

aimodelground runs your data through a configurable **DAG pipeline** with human-in-the-loop gates:

```
ingest → merge → validate → profile → rank_algos
            [GATE: review data]
                        ↓
         train_rf ──┐
         train_xgb ─┤→ eval_join → [GATE: review results] → export → DEPLOY.md
         train_lgb ─┘
```

Each `[GATE]` pauses and waits for your review. Every run is versioned — replay from any node, compare runs, update models with new data.

---

## CLI reference

| Command | Description |
|---------|-------------|
| `aimodelground --version` | Show version |
| `aimodelground init <name>` | Create project |
| `aimodelground run` | Start/resume pipeline |
| `aimodelground run --from <node>` | Replay from node, reuse upstream |
| `aimodelground status` | Show DAG node states |
| `aimodelground approve <node>` | Approve a gate |
| `aimodelground skip <node>` | Skip a node |
| `aimodelground retry <node>` | Reset failed node |
| `aimodelground logs <node>` | Show node logs |
| `aimodelground runs` | List all runs |
| `aimodelground compare <a> <b>` | Diff eval metrics |
| `aimodelground tune` | Optuna hyperparameter search |
| `aimodelground export [--format]` | Re-export model (pickle/onnx) |
| `aimodelground deploy` | Print deployment guide |
| `aimodelground ui [--port N]` | Open web interface |
| `aimodelground features list` | List saved feature sets |
| `aimodelground features info <n>` | Feature set details |
| `aimodelground features delete <n>` | Delete feature set |
| `aimodelground models list` | View all trained models |
| `aimodelground models update [id]` | Update model with new data |

---

## Pipeline configuration (`pipeline.yaml`)

```yaml
nodes:
  - id: ingest_csv
    type: task
    plugin: connectors.file
    config:
      paths: ["data/raw/*.csv"]

  - id: merge
    type: task
    plugin: core.merge
    depends_on: [ingest_csv]

  - id: validate
    type: task
    plugin: validators.schema
    depends_on: [merge]
    config:
      required_columns: [age, income, label]
      max_null_pct: 0.1

  - id: profile
    type: task
    plugin: core.profile
    depends_on: [merge]

  - id: rank_algos
    type: task
    plugin: core.automl_ranker
    depends_on: [profile]

  - id: review_data
    type: gate
    depends_on: [rank_algos, validate]
    message: "Review data before training"

  - id: train_rf
    type: task
    plugin: ml.classical.random_forest
    depends_on: [review_data]
    config:
      target_col: label

  - id: train_xgb
    type: task
    plugin: ml.classical.xgboost
    depends_on: [review_data]
    config:
      target_col: label

  - id: eval_join
    type: parallel_join
    depends_on: [train_rf, train_xgb]

  - id: review_results
    type: gate
    depends_on: [eval_join]
    message: "Review results and pick model"

  - id: export
    type: task
    plugin: core.export
    depends_on: [review_results]
    config:
      format: onnx

  - id: deploy_advisor
    type: task
    plugin: core.deploy_advisor
    depends_on: [export]
```

---

## Data connectors

| Plugin | Source |
|--------|--------|
| `connectors.file` | CSV, JSON, Parquet, Excel, Arrow (DuckDB, glob patterns) |
| `connectors.sql` | PostgreSQL, MySQL, SQLite (SQLAlchemy DSN) |
| `connectors.rest_poll` | HTTP API polling |
| `connectors.websocket` | WebSocket stream |
| `connectors.kafka` | Kafka topic |
| `connectors.image` | PNG/JPG/TIFF directory → image_path + label |
| `connectors.audio` | WAV/MP3/FLAC directory → MFCC features |
| `connectors.s3` | Amazon S3 (DuckDB httpfs, IAM/keys/MinIO) |
| `connectors.gcs` | Google Cloud Storage (DuckDB httpfs) |
| `connectors.feature_store` | Saved feature sets |

---

## ML plugins

### aimodelground-classical

```bash
pip install aimodelground-classical
```

| Plugin | Algorithm | Update support |
|--------|-----------|---------------|
| `ml.classical.random_forest` | RandomForest | warm_start |
| `ml.classical.xgboost` | XGBoost | incremental |
| `ml.classical.lightgbm` | LightGBM | incremental |

All produce: accuracy/F1/RMSE, SHAP feature importance, pickle + ONNX export.

### aimodelground-dl

```bash
pip install aimodelground-dl
```

| Plugin | Architecture |
|--------|-------------|
| `ml.dl.cnn_image` | 3-layer CNN for image classification |
| `ml.dl.lstm_tabular` | 2-layer LSTM for sequential/tabular data |

### aimodelground-llm

```bash
pip install aimodelground-llm
```

| Plugin | Method |
|--------|--------|
| `ml.llm.lora_text` | LoRA fine-tuning on GPT-2, Llama, Mistral, Phi |

---

## Core pipeline plugins

| Plugin | Purpose |
|--------|---------|
| `core.merge` | Concat all connector outputs |
| `core.profile` | Compute DataProfile (row count, column types, nulls) |
| `validators.schema` | Validate required columns + null thresholds |
| `core.automl_ranker` | Rank installed ML plugins by suitability |
| `core.automl_tuner` | Optuna hyperparameter search (CV-based) |
| `core.export` | Export best model (pickle/ONNX/safetensors) |
| `core.deploy_advisor` | Generate DEPLOY.md |
| `core.feature_store_save` | Save processed data as named feature set |
| `core.model_update` | Update existing model with new data |

---

## Feature store

```bash
aimodelground features list
aimodelground features info <name>
aimodelground features versions <name>
aimodelground features delete <name>
```

```yaml
# Save features in pipeline
- id: save_features
  type: task
  plugin: core.feature_store_save
  depends_on: [merge]
  config:
    feature_name: customer_features_v1

# Load in future run
- id: load_features
  type: task
  plugin: connectors.feature_store
  config:
    name: customer_features_v1
```

---

## Model update

```bash
aimodelground models list
aimodelground models update --data data/raw/new.csv --target label
aimodelground models update run_001/random_forest --n-estimators 100
```

---

## Versioned runs

```bash
aimodelground runs
aimodelground compare run_001 run_002
aimodelground run --from validate    # replay, reuse upstream outputs
```

---

## Web UI

```bash
aimodelground ui --port 8765
```

- **Pipeline** — live DAG, approve/skip buttons, SSE real-time updates
- **Data** — file upload, schema, null stats
- **Results** — leaderboard, Plotly charts, run comparison
- **Deploy** — rendered deployment guide

---

## Project structure

```
my-project/
  pipeline.yaml         # DAG definition
  project.db            # SQLite state
  data/raw/             # Input data
  runs/
    run_001/
      artifacts/        # Models, parquets, ranking.json
      logs/             # Node logs
      eval_report.json
      DEPLOY.md         # Deployment guide
      export/           # Exported model
  .modelbuilder/
    features/           # Feature store data
    feature_store.db
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Releasing

See [RELEASING.md](RELEASING.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Apache 2.0 — see [LICENSE](LICENSE)




