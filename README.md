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

Every step is a **node** in the DAG. Gates pause execution and wait for your approval. You can use the **CLI** (terminal-first) or the **Web UI** (browser-first) — both share the same project state.

---

## Step-by-step usage

### Step 1 — Create a project

```bash
aimodelground init my-churn-model
cd my-churn-model
```

This creates:

```
my-churn-model/
  pipeline.yaml      ← DAG definition (edit this)
  data/raw/          ← drop your data files here
  .modelbuilder/     ← project config
```

---

### Step 2 — Add your data

Drop any supported file into `data/raw/`:

```bash
cp customers.csv my-churn-model/data/raw/
# or: .parquet, .json, .xlsx, .png folder, .wav folder
```

For SQL databases, S3, GCS, Kafka, REST APIs — configure the connector in `pipeline.yaml` (see [Data connectors](#data-connectors)).

---

### Step 3 — Configure `pipeline.yaml`

Open `pipeline.yaml`. The default template is pre-filled. You only need to set **two things**:

**a) Point to your data:**

```yaml
- id: ingest
  type: task
  plugin: connectors.file
  config:
    paths: ["data/raw/customers.csv"]   # ← your file
```

**b) Set your target column** (the column you want to predict):

```yaml
- id: train_rf
  type: task
  plugin: ml.classical.random_forest
  depends_on: [review_data]
  config:
    target_col: churn    # ← column name to predict
```

Everything else (merge, validate, profile, rank, eval, export) runs automatically.

---

### Step 4 — Run the pipeline

**Using the CLI:**

```bash
aimodelground run
```

The pipeline starts. It will run until it hits the first review gate, then print:

```
GATE: review_data
   Review data profile and algorithm rankings before training
   Run: aimodelground approve review_data
```

**Using the Web UI:**

```bash
aimodelground ui
# Opens http://localhost:8765 in your browser
```

The **Pipeline** tab shows each node with a live status indicator. Nodes turn green as they complete.

---

### Step 5 — Check what the pipeline found (first gate)

Before training starts, aimodelground profiles your data and ranks algorithms. Review what it discovered:

**CLI:**

```bash
aimodelground status
```

Output:

```
Pipeline: my-churn-model  run_001  4/8 nodes done

  +  ingest          succeeded
  +  merge           succeeded
  +  validate        succeeded
  +  profile         succeeded
  +  rank_algos      succeeded
  ?  review_data     AWAITING  → aimodelground approve review_data
  .  train_rf        pending
  .  train_xgb       pending
```

To see the full data profile and algorithm rankings:

```bash
# Check the profile saved in the run artifacts
cat runs/run_001/artifacts/profile.json

# Check which algorithms were ranked and why
cat runs/run_001/artifacts/ranking.json
```

**Web UI:** The **Data** tab shows your column types, null counts, and distributions. The Pipeline tab shows the ranking results inline on the `rank_algos` node.

If the data looks wrong (wrong types, too many nulls, wrong file loaded) — fix the issue and retry:

```bash
aimodelground retry ingest   # re-runs ingest and all downstream nodes
aimodelground run            # resumes
```

If everything looks good — approve the gate:

```bash
aimodelground approve review_data
```

**Web UI:** Click the **Approve** button on the `review_data` gate node.

Then resume:

```bash
aimodelground run
```

---

### Step 6 — Wait for training

Training runs in parallel for all selected algorithms. Watch progress:

**CLI:**

```bash
aimodelground status          # check node states
aimodelground logs train_rf   # tail logs for a specific node
```

**Web UI:** The Pipeline tab updates live. Click any running node to see its log output in the side panel.

Training time depends on your data size and hardware:
- Tabular data, 10k–100k rows: typically 30 seconds – 5 minutes
- Images / sequences: minutes to hours depending on GPU

---

### Step 7 — Review results (second gate)

After all models finish, the pipeline pauses again:

**CLI:**

```bash
aimodelground status
# shows: review_results  AWAITING

# View the eval report
cat runs/run_001/eval_report.json
```

**Web UI:** Go to the **Results** tab. You'll see:
- Leaderboard table: each algorithm with accuracy, F1, RMSE
- Feature importance chart (SHAP values)
- Option to compare against a previous run

If results are poor:
- Try tuning hyperparameters first: `aimodelground tune --trials 50`
- Or re-run with different data: `aimodelground run --from ingest`
- Or skip a poorly-performing algorithm: `aimodelground skip train_xgb`

When satisfied — approve:

```bash
aimodelground approve review_results
aimodelground run
```

**Web UI:** Click **Approve** on the `review_results` gate.

---

### Step 8 — Export and deploy

After approval, the pipeline exports the best model and generates `DEPLOY.md`.

**CLI:**

```bash
aimodelground deploy
# Prints the full deployment guide with code examples
```

**Web UI:** Go to the **Deploy** tab. It shows:
- Model info (algorithm, format, input schema)
- Python inference script
- FastAPI REST endpoint (copy-paste ready)
- Dockerfile

By default the model exports as `pickle`. To export as ONNX:

```yaml
# in pipeline.yaml
- id: export
  type: task
  plugin: core.export
  depends_on: [review_results]
  config:
    format: onnx     # or: pickle, safetensors
```

Or re-export after the fact:

```bash
aimodelground export --format onnx
```

The exported file is at `runs/run_001/export/model.onnx` (or `.pkl`).

---

### Step 9 — Iterate

**Compare two runs:**

```bash
aimodelground compare run_001 run_002
```

Output:

```
Comparing run_001 vs run_002
 Metric    run_001    run_002    Delta
 accuracy  0.8412     0.8891    +0.0479
 f1        0.8103     0.8654    +0.0551
```

**Replay from a specific node** (e.g., re-train with different config without re-ingesting):

```bash
# Edit pipeline.yaml — change n_estimators, learning_rate, etc.
aimodelground run --from train_rf
```

**Update an existing model with new data:**

```bash
aimodelground models list
aimodelground models update run_001/random_forest --data data/raw/new_customers.csv
```

---

### Common issues

| Problem | Fix |
|---------|-----|
| Node shows `failed` | `aimodelground logs <node>` to see error. Fix the issue, then `aimodelground retry <node>` |
| Wrong target column | Edit `pipeline.yaml`, set correct `target_col`, then `aimodelground run --from train_rf` |
| Too many nulls in data | Fix source data, then `aimodelground retry ingest` |
| Training too slow | Reduce dataset size for prototyping, or add GPU. For tabular data, `n_estimators: 50` trains faster |
| Model accuracy too low | Run `aimodelground tune --trials 100` before the training gate, or add more data |
| Want to skip an algorithm | `aimodelground skip train_xgb` — downstream nodes unblock automatically |
| Web UI not updating | Check `aimodelground run` is still running in another terminal |

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




