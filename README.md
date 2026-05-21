# aimodelground

[![PyPI version](https://img.shields.io/pypi/v/aimodelground.svg)](https://pypi.org/project/aimodelground/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-231%20passing-brightgreen.svg)](#)

**Privacy-first, locally-installed ML model builder.**

Upload data from any source, let the app guide you step-by-step through training, and get a deployable model — entirely on your machine. No cloud, no telemetry, no data leaving your system.

---

## Installation

```bash
pip install aimodelground
```

**Upgrading from a previous version:**

```bash
# Upgrade to latest
pip install --upgrade aimodelground

# Pin to a specific version
pip install "aimodelground==0.3.0"
```

> **Note:** `pip install aimodelground` without flags will print "Requirement already satisfied" if any version is already installed and will NOT upgrade. Use `--upgrade` or pin the version explicitly.

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

## Using the CLI — step by step

The CLI is the primary interface. Every action is a single command.

### 1. Create a project

```bash
aimodelground init my-project
cd my-project
```

Creates `pipeline.yaml`, `data/raw/`, `.modelbuilder/config.yaml`.

---

### 2. Add your data

```bash
cp customers.csv data/raw/
# or: .parquet, .json, .xlsx, .pdf, .docx
```

---

### 3. Configure the pipeline

Open `pipeline.yaml` and set:

```yaml
- id: ingest
  plugin: connectors.file
  config:
    paths: ["data/raw/customers.csv"]   # ← your file

- id: train_rf
  plugin: ml.classical.random_forest
  config:
    target_col: churn                   # ← column to predict
```

---

### 4. Start the pipeline

```bash
aimodelground run
```

Runs until the first gate, prints what to do next.

---

### 5. Check progress

```bash
aimodelground status
```

```
  +  ingest          succeeded
  +  profile         succeeded
  ?  review_data     AWAITING  → aimodelground approve review_data
  .  train_rf        pending
```

---

### 6. Review data, then approve

```bash
# See what the profile and algorithm ranking found
cat runs/run_001/artifacts/profile.json
cat runs/run_001/artifacts/ranking.json

# Happy with data quality? Approve the gate
aimodelground approve review_data

# Resume
aimodelground run
```

If anything is wrong: `aimodelground retry ingest` to re-run from ingestion.

---

### 7. Wait for training, then review results

```bash
aimodelground status          # watch node states
aimodelground logs train_rf   # tail training log

# Once eval_join completes, review metrics
cat runs/run_001/eval_report.json

# Optionally tune hyperparameters before approving
aimodelground tune --trials 50

# Approve
aimodelground approve review_results
aimodelground run
```

---

### 8. Get deployment guide

```bash
aimodelground deploy
```

Prints the full `DEPLOY.md` with Python script, FastAPI endpoint, and Dockerfile.

---

### 9. Iterate

```bash
aimodelground runs                        # list all runs
aimodelground compare run_001 run_002     # diff metrics
aimodelground run --from train_rf         # re-train with new config
aimodelground models update               # update model with new data
aimodelground export --format onnx        # re-export in different format
```

---

## Using the Web UI — step by step

The Web UI is a guided 6-step wizard. From v0.3.0 you can run the entire pipeline (upload → train → deploy → query) without touching the terminal.

```bash
cd my-project
aimodelground ui
# Opens http://localhost:8765
```

The wizard stepper at the top tracks your progress. Completed steps are clickable (green ✓). Steps unlock as you complete each stage.

```
 ✓ Upload  →  ✓ Configure  →  ▶ Run  →  · Results  →  · Deploy  →  · Query
```

---

### Step 1 — Upload

Drag and drop your data file, or click the upload zone to browse.

```
┌─────────────────────────────────────────────────────────┐
│  Upload Data                                            │
│  Drop a file to get started — CSV, JSON, Parquet...    │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐  │
│  │                                                  │  │
│  │              ⇩  Drop file here                   │  │
│  │         or click to browse                       │  │
│  │    CSV · JSON · Parquet · Excel · PDF · DOCX     │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  Files in data/raw/  (1 file)                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 📄 iris.csv               24.1 KB      ready     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Files land in `data/raw/`. Move to Configure once your file appears in the list.

---

### Step 2 — Configure

The left pane auto-detects your file's columns. The right pane shows live YAML that updates as you change the form.

```
┌─────────────────────────────────────────────────────────────────────┐
│  pipeline.yaml                              [Validate]  [Save]      │
├──────────────────────────────┬──────────────────────────────────────┤
│  DATA FILE                   │  Live YAML                           │
│  ▾ iris.csv                  │  nodes:                              │
│    150 rows · 5 cols         │    - id: ingest_files                │
│                              │      plugin: connectors.file         │
│  TARGET COLUMN               │      config:                         │
│  ▾ species (categorical)     │        paths: ["data/raw/iris.csv"]  │
│                              │        target_col: "species"         │
│  ALGORITHMS                  │                                      │
│  [✓ RandomForest] [✓ XGBoost]│    - id: validate                   │
│  [ LightGBM    ] [ LSTM    ] │      plugin: validators.schema       │
│                              │      depends_on: [ingest_files]      │
│  TASK TYPE                   │                                      │
│  [✓ Classification] [Regress]│    - id: review_data                 │
│                              │      type: gate                      │
└──────────────────────────────┴──────────────────────────────────────┘
```

You can edit the YAML directly too — form and YAML stay in sync. Click **Save** when done.

---

### Step 3 — Run

Click **Run Pipeline** — no terminal needed. The pipeline runs in the background with live node updates.

```
┌─────────────────────────────┐  ┌─────────────────────────────────┐
│  Pipeline Control           │  │  Nodes                          │
│                             │  │                                 │
│  [▶ Run Pipeline] [From: ▾] │  │  ▓ DONE   ingest_files         │
│                             │  │           connectors.file       │
│  Progress                   │  │                                 │
│  ████████░░░░░░  3/8 nodes  │  │  ▓ DONE   validate             │
│                             │  │           validators.schema     │
│  ┌─────────────────────┐   │  │                                 │
│  │ ⏳ Gate: review_data│   │  │  ⏳ GATE  review_data           │
│  │ Review data profile  │   │  │           awaiting approval     │
│  │ before training.    │   │  │                                 │
│  │ [✓ Approve] [Skip]  │   │  │  ·  PEND  profile              │
│  └─────────────────────┘   │  │  ·  PEND  rank_algos           │
│                             │  │  ·  PEND  export_model         │
└─────────────────────────────┘  └─────────────────────────────────┘
```

Gate cards appear automatically for nodes that need your review. Click **Approve** to continue — the pipeline resumes without restarting.

---

### Step 4 — Results

Metric summary cards at the top, feature importance bars below. Compare runs side by side.

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   94.20%     │  │    0.9412    │  │    0.9780    │
│   ACCURACY   │  │   F1 SCORE   │  │     AUC      │
│  ↑ +2.1%     │  │  ↑ +0.018   │  │  — baseline  │
└──────────────┘  └──────────────┘  └──────────────┘

Feature Importance (SHAP)
  petal_length  ████████████████████████████  0.912
  petal_width   ████████████████████          0.782
  sepal_length  ████████████                  0.421
  sepal_width   ██████                        0.213
```

Switch between runs using the selector at the top. Click **vs run_001** to diff two runs with coloured deltas (green = improvement).

---

### Step 5 — Deploy

Auto-generated deployment guide with copy buttons. Links directly to the Query step.

```
┌────────────────────────────────────┐  ┌─────────────────────┐
│  DEPLOY.md — run_003    [Copy]     │  │  Export Info        │
│                                    │  │  Algorithm: RF      │
│  ## Option 1 — Python             │  │  Format:  pickle    │
│                                    │  │  runs/.../model.pkl │
│  import joblib                     │  │  [Copy path]        │
│  model = joblib.load("model.pkl")  │  ├─────────────────────┤
│  pred = model.predict([features])  │  │  Quick Actions      │
│                                    │  │  [Query Model →]    │
│  ## Option 2 — FastAPI            │  │  [View Metrics]     │
│  ...                               │  │  [Back to Pipeline] │
└────────────────────────────────────┘  └─────────────────────┘
```

---

### Step 6 — Query

Two tabs: **Predict** (run inference) and **Explain** (SHAP insights). No external API or LLM required — everything runs locally from your exported model.

**Predict tab** — type feature values and get an instant prediction:

```
┌──────────────────────────────────────────────────┐
│  🎯 Predict  |  🔍 Explain                       │
├──────────────────────────────────────────────────┤
│  Enter feature values                            │
│                                                  │
│  sepal_length  [5.1    ]   sepal_width  [3.5   ] │
│  petal_length  [1.4    ]   petal_width  [0.2   ] │
│                                                  │
│  [Predict →]  [Clear]                            │
│                                                  │
│  ┌─────────────────────────────────────────────┐ │
│  │  setosa                  Confidence: 99%    │ │
│  │  Top driver: petal_length = 1.4             │ │
│  └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

**Explain tab** — reads SHAP values, metrics, and profile from run artifacts:

```
METRICS
  accuracy     0.9420
  f1           0.9412

FEATURE IMPORTANCE (SHAP)
  petal_length  ████████████████████  0.912
  petal_width   ████████████████      0.782

INSIGHTS
  💡 'petal_length' dominates predictions (score 0.91) — model may overfit.
```

---

### Theme

The UI ships with a **Deep Space dark theme** and supports light mode. Click the ☀ Light button in the top bar to toggle — preference is saved in `localStorage`.

```
┌─────────────────────────────────────────────────────────┐
│  model-builder  v0.3.0     ● live  my-project  ☀ Light │
│ ─────────────────────────────────────────────────────── │
│  ✓ Upload  →  ✓ Configure  →  ▶ Run  →  · Results ...  │
└─────────────────────────────────────────────────────────┘
```

Dark (default): `#0a0e1a` background, `#4f8ef7` accent. Light: white background, `#2563eb` accent.

---

## Step-by-step usage (combined reference)

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

**Using the Web UI (recommended):** Go to the **Configure** step. The form auto-detects your file's columns and pre-fills the target column dropdown. Select your target, choose algorithms, and click **Save** — the YAML is written for you.

**Using the CLI:** Open `pipeline.yaml`. The default template is pre-filled. You only need to set **two things**:

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

Go to the **Run** step (step 3 in the wizard). Click **Run Pipeline** — the pipeline starts immediately, no terminal needed. Nodes update live as they complete.

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
| `connectors.document` | **PDF, DOCX, TXT, MD** — extracts text, page numbers, char count |
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

## Working with PDF and document files

If your data is PDFs, Word documents, text files, or markdown, use `connectors.document`. It extracts text from each file (page-by-page for PDFs) and produces a DataFrame with `filename`, `text`, `page`, and `char_count` columns.

### Step 1 — Organise your files

**Option A — flat folder** (all documents, no labels):
```
data/raw/
  contract_001.pdf
  contract_002.pdf
  report_march.docx
  notes.txt
```

**Option B — labelled subdirectories** (for classification):
```
data/raw/
  approved/
    doc_001.pdf
    doc_002.pdf
  rejected/
    doc_003.pdf
    doc_004.pdf
```

### Step 2 — Configure `pipeline.yaml`

```yaml
nodes:
  - id: ingest_docs
    type: task
    plugin: connectors.document
    config:
      paths: ["data/raw/**/*.pdf", "data/raw/**/*.docx"]
      label_from_dir: true   # set true if using labelled subdirectories

  - id: merge
    type: task
    plugin: core.merge
    depends_on: [ingest_docs]

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
    depends_on: [rank_algos]
    message: "Review extracted text before training"

  - id: train_lora
    type: task
    plugin: ml.llm.lora_text
    depends_on: [review_data]
    config:
      text_col: text          # column produced by the document connector
      label_col: label        # column from label_from_dir, or your own label column
      base_model: gpt2        # or: meta-llama/Llama-2-7b, mistralai/Mistral-7B-v0.1
      epochs: 3
      max_length: 512

  - id: review_results
    type: gate
    depends_on: [train_lora]
    message: "Review fine-tuning results"

  - id: export
    type: task
    plugin: core.export
    depends_on: [review_results]
    config:
      format: safetensors     # adapter weights, compatible with Ollama / vLLM

  - id: deploy_advisor
    type: task
    plugin: core.deploy_advisor
    depends_on: [export]
```

### Step 3 — Run

```bash
pip install aimodelground-llm   # required for LLM fine-tuning

aimodelground run
```

The connector extracts text from every PDF/DOCX, then the LLM plugin fine-tunes a LoRA adapter on your labelled documents.

### What the extracted data looks like

| filename | source | page | total_pages | text | char_count | label |
|----------|--------|------|-------------|------|------------|-------|
| contract_001.pdf | data/raw/approved/... | 1 | 4 | "This agreement..." | 3420 | approved |
| contract_001.pdf | data/raw/approved/... | 2 | 4 | "Section 2..." | 2870 | approved |

Each PDF produces one row per page. DOCX and TXT produce one row per file.

### Choosing a base model

| Base model | When to use | GPU required |
|-----------|-------------|-------------|
| `gpt2` | Small datasets (<1000 docs), fast iteration, CPU-friendly | No (CPU works) |
| `distilbert-base-uncased` | Classification tasks, small model, good accuracy | No |
| `meta-llama/Llama-2-7b` | Large datasets, high accuracy, production use | Yes (8GB+ VRAM) |
| `mistralai/Mistral-7B-v0.1` | Best accuracy, multilingual support | Yes (8GB+ VRAM) |

### Mixing documents with other data

You can combine document text with structured data in the same pipeline:

```yaml
nodes:
  - id: ingest_docs
    type: task
    plugin: connectors.document
    config:
      paths: ["data/raw/contracts/**/*.pdf"]
      label_from_dir: true

  - id: ingest_metadata
    type: task
    plugin: connectors.file
    config:
      paths: ["data/raw/contract_metadata.csv"]

  - id: merge
    type: task
    plugin: core.merge
    depends_on: [ingest_docs, ingest_metadata]
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

6-step wizard. No terminal needed for basic use from v0.3.0.

| Step | URL | What it does |
|------|-----|-------------|
| **Upload** | `/upload` | Drag-drop data files, see file list |
| **Configure** | `/configure` | Smart form + live YAML editor, auto-detects columns |
| **Run** | `/` | Run button, live node list, gate approval, progress bar |
| **Results** | `/results` | Metric cards, SHAP bars, Plotly chart, run comparison |
| **Deploy** | `/deploy` | Deployment guide, export info, copy buttons |
| **Query** | `/query` | Predict tab (model inference) + Explain tab (SHAP + insights) |

Dark theme default, light mode toggle (preference stored in browser). See the [Web UI walkthrough](#using-the-web-ui--step-by-step) above for screenshots.

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





