# Spam / Ham Email Classifier — Production MLOps System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-8A2BE2)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Hydra](https://img.shields.io/badge/Hydra-OmegaConf-89b4fa)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.x-017CEE?logo=apacheairflow&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-K8s-326CE5?logo=kubernetes&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Grafana-E6522C?logo=prometheus&logoColor=white)
![CI](https://img.shields.io/github/actions/workflow/status/your-org/fake-news-detection-ml-system/ci.yml?label=CI&logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end, production-grade NLP pipeline for binary email classification (spam vs. ham). Built on the [TREC 2006 Public Corpus](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2006), the system covers the full ML lifecycle — from experiment tracking and model registry to Kubernetes deployment, automated retraining, and real-time observability.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [ML Pipeline & Experiments](#ml-pipeline--experiments)
  - [Dataset & EDA](#dataset--eda)
  - [Model Selection](#model-selection)
  - [Fine-Tuning Strategy (LoRA)](#fine-tuning-strategy-lora)
  - [Final Metrics](#final-metrics)
- [Configuration System (Hydra + OmegaConf)](#configuration-system-hydra--omegaconf)
- [Training](#training)
- [Inference SDK](#inference-sdk)
- [API](#api)
  - [REST Endpoint](#rest-endpoint)
  - [Telegram Bot](#telegram-bot)
  - [Rate Limiting & Security](#rate-limiting--security)
- [Observability (Prometheus + Grafana)](#observability-prometheus--grafana)
- [MLflow Model Registry](#mlflow-model-registry)
- [Airflow DAGs](#airflow-dags)
- [Kubernetes Deployment](#kubernetes-deployment)
- [CI/CD](#cicd)
- [Demo](#demo)
- [Quickstart (Local)](#quickstart-local)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│   Kaggle Dataset (TREC 2006)  ──►  TextCleaningPipeline          │
│   (HTML strip, non-printable regex, whitespace normalization)     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                      TRAINING PIPELINE                            │
│   PyTorch Lightning  +  PEFT / LoRA  +  Hydra config             │
│   MLflow Experiment Tracking  ──►  MLflow Model Registry          │
│   Checkpoints: val_f1 monitored, EarlyStopping (patience=3)       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                   MLFLOW MODEL REGISTRY                           │
│   Staging alias  ──►  (manual gate)  ──►  Production alias        │
│   Airflow DAG: promote_to_prod  (schedule=None, manual trigger)   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                        SERVING LAYER                              │
│   FastAPI (uvicorn)  +  Telegram Bot (aiogram, webhook mode)      │
│   NLPPipeline SDK  ──►  /api/v1/classify                          │
│   API Key auth  +  SlowAPI rate limiting (20 req/min)             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                     OBSERVABILITY STACK                           │
│   Prometheus (scrape: /metrics, k8s pod autodiscovery)            │
│   Grafana dashboards  +  kube-state-metrics  +  node-exporter     │
│   Custom metrics: nlp_classification_requests_total,              │
│                   nlp_classification_inference_seconds            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                      ORCHESTRATION                                │
│   Apache Airflow (KubernetesPodOperator)                          │
│   weekly_classifier_finetuning  ──►  evaluate_staging             │
│   ──►  Slack notification  ──►  promote_to_prod (manual)          │
│   + batch_analytics (daily)  +  quality_control (weekly)          │
│   + system_maintenance                                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── configs/                        # Hydra config tree (single source of truth)
│   ├── main.yaml                   # Root config — assembles all defaults
│   ├── model/
│   │   ├── architecture/           # bert-base-multilingual, rubert, bge-reranker, etc.
│   │   └── finetuning/             # full.yaml | head_only.yaml | lora.yaml
│   ├── data/
│   │   └── source/                 # kaggle_dataset | hf_dataset | local_csv
│   ├── environment/                # local.yaml | prod.yaml
│   ├── api/                        # FastAPI + Telegram webhook config
│   ├── trainer/default.yaml        # PyTorch Lightning Trainer
│   ├── logger/mlflow.yaml          # MLflow tracking URI, experiment name
│   └── prometheus.yml              # Prometheus scrape config
│
├── src/
│   ├── train.py                    # Training entrypoint (Hydra @main)
│   ├── eval.py                     # Evaluation entrypoint
│   ├── infer.py                    # Single-shot CLI inference
│   ├── tune.py                     # Optuna hyperparameter search
│   ├── run_api.py                  # API entrypoint (uvicorn)
│   ├── core/
│   │   ├── data/                   # DataModule, Dataset, Fetcher, Cleaners, Collators
│   │   └── models/                 # HFModelBuilder, HFTokenizerBuilder
│   ├── training/
│   │   ├── module.py               # LightningModule: loss, metrics, optimizer
│   │   └── tuner.py                # Optuna integration
│   ├── api/
│   │   ├── rest/                   # FastAPI router, endpoints, middlewares, rate limiter
│   │   ├── tg_bot/                 # aiogram bot (webhook + local polling modes)
│   │   ├── metrics.py              # Prometheus Counter + Histogram definitions
│   │   └── schemas.py              # Pydantic request/response models
│   ├── sdk/
│   │   └── inference.py            # NLPPipeline: production-ready inference class
│   ├── jobs/                       # Airflow job scripts (promote, analytics, maintenance)
│   └── utils/                      # Hydra utils, MLflow pip requirements, torch helpers
│
├── dags/                           # Airflow DAG definitions
│   ├── retrain_model_dag.py        # Weekly finetuning + staging eval + Slack gate
│   ├── promote_to_prod.py          # Manual promotion: Staging → Production + rollout
│   ├── batch_analytics.py          # Daily analytics (LLM-as-an-Analyst)
│   ├── quality_control.py          # Weekly drift detection
│   └── system_maintenance.py       # Infra housekeeping
│
├── deploy/
│   ├── k8s/                        # Kubernetes manifests (Deployment, Service, Ingress,
│   │   │                           #   PVCs, RBAC, Secrets, monitoring exporters)
│   │   └── monitoring/             # kube-state-metrics, node-exporter
│   └── airflow/variables.json      # Airflow Variables bootstrap
│
├── notebooks/                      # Research notebooks (EDA, baseline, PEFT sandbox)
│   ├── 01_eda_and_tokens.ipynb
│   ├── 02_classification_baseline.ipynb
│   ├── 04_evaluation_metrics.ipynb
│   └── 05_peft_lora_sandbox.ipynb
│
├── tests/
│   ├── api/test_classifier.py      # FastAPI endpoint tests (pytest-asyncio)
│   └── dags/test_dag_config.py     # DAG structure / import tests
│
├── demo/
│   ├── app.py                      # Streamlit demo UI
│   ├── Dockerfile                  # Standalone demo image
│   ├── requirements.txt
│   └── quickstart.ipynb
│
└── .github/workflows/ci.yml        # GitHub Actions: lint (ruff) + pytest
```

---

## ML Pipeline & Experiments

### Dataset & EDA

**Dataset:** [`bayes2003/emails-for-spam-or-ham-classification-trec-2006`](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2006)

**Raw data quality audit:**

| Issue | Rate |
|---|---|
| HTML tags | 0.00% |
| Extra whitespace | 0.00% |
| Truncated lines | 0.00% |
| Non-printable characters | 0.06% |

Non-printable characters are removed via a regex cleaner in the data pipeline (`pattern="[^\x20-\x7E\n\t]"`). All cleaners are declared in `configs/data/default.yaml` and instantiated by Hydra.

**Sequence length analysis:**

- Median token length: **280 tokens**
- At `max_length=512`: ~26.5% of examples are truncated
- Working config uses `max_length=256` — a deliberate speed/quality trade-off confirmed by ablation experiments (see below)

**Class imbalance:**

| Class | Share |
|---|---|
| Majority (ham) | 72.3% |
| Minority (spam) | 27.7% |

Class imbalance is addressed via automatic inverse-frequency weighting (`class_weights: "auto"` in `configs/data/default.yaml`). This alone produced a significant quality jump:

| Metric | Without weights | With weights |
|---|---|---|
| Recall | 0.667 | **0.778** |
| Val F1 | 0.706 | **0.922** |

---

### Model Selection

Four encoder architectures were benchmarked under identical conditions (3 epochs, `lr=1e-3`, class weights enabled, `head_only` fine-tuning):

| Model | Train time | Test F1 | Test Acc | Test Loss | Expected Recall |
|---|---|---|---|---|---|
| **deepset/bert-base-cased-squad2** *(selected)* | 3.4 min | **0.700** | **0.694** | **0.590** | 0.421 |
| bert-base-multilingual-cased | 3.1 min | 0.689 | 0.672 | 0.590 | 0.421 |
| BAAI/bge-reranker-v2-m3 | 12.4 min | 0.610 | 0.606 | 0.617 | 0.526 |
| dslim/bert-base-NER | 3.0 min | 0.620 | 0.693 | 0.636 | **0.053** |

> **Why the winning model was chosen:**
> The selected encoder achieves the best F1 and lowest loss with a comfortable 3.4-minute training time. The cross-lingual model spreads vocabulary across 104 languages, reducing per-language quality. The reranker (`bge-reranker-v2-m3`) is 4× slower with worse accuracy — architecturally oversized for binary classification. The NER model has deeply task-specific encoder weights that do not transfer well to sequence classification, evidenced by a near-zero Expected Recall (0.053).

The architecture is swappable via a single config flag — no code changes required:

```bash
python -m src.train model/architecture=google_bert_mulitling
```

---

### Fine-Tuning Strategy (LoRA)

**Context length ablation** (frozen backbone, 3 epochs, 50 batch limit):

| max_length | Val F1 | Expected Recall | Train time |
|---|---|---|---|
| 256 tokens | 0.615 | **0.421** | 3.4 min |
| 512 tokens | **0.687** | 0.263 | 6.6 min |

256 tokens was selected: it keeps recall high and fits comfortably in a local 8 GB RAM container.

**LoRA adapter selection** (overfitting probe — 1 batch, 100 steps, `lr=1e-3`):

All tested target module combinations (`query+value`, `query+key+value`, `query+value+dense`, `query+key+value+dense`) converged to Train Acc = 1.0 with End Loss = 0.0, confirming sufficient capacity. The lightest configuration was chosen following Occam's Razor:

```yaml
# configs/model/finetuning/lora.yaml
type: "peft"
peft_config:
  r: 8
  lora_alpha: 8
  target_modules: ["query", "value"]
  lora_dropout: 0.1
  task_type: "SEQ_CLS"
```

**Trainable parameters: 0.1664% of total model weights.** LoRA adapters are merged (`merge_and_unload`) before registration in MLflow, so the production model has zero PEFT runtime dependency.

The fine-tuning mode is also switchable via config with no code changes:

```bash
python -m src.train model/finetuning=full      # Full fine-tune
python -m src.train model/finetuning=head_only # Classifier head only
python -m src.train model/finetuning=lora      # LoRA (default)
```

---

### Final Metrics

> Results obtained under resource-constrained local conditions (`limit_train_batches: 50`, `max_length: 256`, CPU/MPS). Production runs on GPU with the full dataset are expected to yield higher numbers.

| Metric | Value |
|---|---|
| Training time (local) | 6.1 min |
| **Val F1 / Accuracy** | **0.9655** |
| **Test F1 / Accuracy** | **0.9354** |
| **Expected Recall** | **0.8947** |
| Optimal classification threshold | 0.5422 |

The precision/recall trade-off is configurable without retraining — adjust `target_precision` and `target_recall` in `configs/main.yaml` to shift the decision threshold at inference time.

---

## Configuration System (Hydra + OmegaConf)

All components — model architecture, fine-tuning strategy, data source, training hyperparameters, and API settings — are declared in `configs/`. Hydra composes them at runtime with full override support.

**Key override examples:**

```bash
# Switch to production environment (full dataset, GPU, bf16)
python -m src.train environment=prod

# Change data source to local CSV
python -m src.train data/source=local_csv

# Override individual hyperparameters
python -m src.train trainer.max_epochs=10 model_module.optimizer_cfg.lr=2e-4

# Full production training run
python -m src.train environment=prod model/finetuning=lora model/architecture=rubert_base
```

**Environment configs** handle the local/prod split cleanly:

| Parameter | `environment=local` | `environment=prod` |
|---|---|---|
| `accelerator` | auto (CPU/MPS/GPU) | gpu |
| `precision` | 32-true | 16-mixed |
| `batch_size` | 4 | 16 |
| `num_workers` | 0 | 4 |
| `limit_train_batches` | configurable | 1.0 (full) |
| `max_epochs` | 10 | 10 |

---

## Training

```bash
# Install dependencies
uv pip install -e ".[train]"

# Local quick run (CPU, debug mode)
python -m src.train

# Production training (GPU, full dataset)
python -m src.train environment=prod

# Hyperparameter search (Optuna)
python -m src.tune

# Evaluate a checkpoint
python -m src.eval ckpt_path=./models/best.ckpt
```

After training completes:
1. Best checkpoint is selected by `val_f1` (ModelCheckpoint callback).
2. LoRA adapters are merged into the base model (`merge_and_unload`).
3. Model is logged to MLflow as a `transformers` flavor with explicit pip requirements.
4. Version is tagged `Staging`. Promotion to `Production` requires a manual DAG trigger.

---

## Inference SDK

`NLPPipeline` is the single inference interface used by both the REST API and the Telegram bot. It handles Hydra config loading, model/tokenizer initialization, MLflow registry lookup, and optional checkpoint loading transparently.

```python
from src.sdk.inference import NLPPipeline

pipeline = NLPPipeline(config_name="main")  # loads Production alias from MLflow

results = pipeline(["Free money! Click here now!", "Meeting notes from Tuesday."])
# [{'label_id': 1, 'confidence': 0.9871, 'all_probabilities': [0.0129, 0.9871]},
#  {'label_id': 0, 'confidence': 0.9954, 'all_probabilities': [0.9954, 0.0046]}]
```

The classification threshold is read from `configs/main.yaml` (`inference.threshold`), making it adjustable without rebuilding the image.

---

## API

### REST Endpoint

```
POST /api/v1/classify
Header: X-API-Key: <key>

{
  "text": "Congratulations! You've been selected for a free prize."
}
```

Response:

```json
{
  "label_id": 1,
  "confidence": 0.9871,
  "all_probabilities": [0.0129, 0.9871]
}
```

Additional endpoints:

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /metrics` | Prometheus metrics scrape |

### Telegram Bot

The bot runs in two modes depending on environment:

- **Production** (`bot_webhook.py`): Receives updates via the `/webhook/telegram` FastAPI endpoint. The ML model is called in-process (no HTTP hop), with inference offloaded to a thread pool (`asyncio.to_thread`) to avoid blocking the event loop.
- **Local development** (`bot_local.py`): Long-polling mode. Calls the REST API over HTTP, enabling the bot to run independently from the server process.

### Rate Limiting & Security

- API key authentication via `X-API-Key` header (configurable via `API_KEY` env var).
- SlowAPI rate limiter: **20 requests/minute** per client IP on `/api/v1/classify`.
- CORS origins are declared in `configs/api/fastapi.yaml`.

---

## Observability (Prometheus + Grafana)

Custom application metrics are defined in `src/api/metrics.py`:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `nlp_classification_requests_total` | Counter | `source` (rest / tg) | Total classification calls |
| `nlp_classification_inference_seconds` | Histogram | `source` (rest / tg) | Model inference latency |

Prometheus scrapes `/metrics` (exposed by `prometheus-fastapi-instrumentator`) on port 8000.

**Scrape targets configured in `configs/prometheus.yml`:**
- FastAPI service (local Docker Compose: `fake_news_api_server:8000`)
- Kubernetes pods with `prometheus.io/scrape: "true"` annotation (auto-discovery)
- `kube-state-metrics` — cluster resource state
- `node-exporter` — host-level hardware metrics

The K8s Deployment manifest includes the required Prometheus annotations:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

---

## MLflow Model Registry

Models flow through a two-stage registry:

```
Training run  ──►  Staging alias  ──►  (manual approval)  ──►  Production alias
```

The `NLPPipeline` SDK loads the `Production` alias automatically. No image rebuild is required when promoting a new model version — the running pod picks up the new weights on the next request (or after `kubectl rollout restart` triggered by the `promote_to_prod` DAG).

To manually promote a staged model:

```bash
# Via Airflow UI: trigger DAG "promote_to_prod"
# Or directly via MLflow client:
python -m src.jobs.promote
```

---

## Airflow DAGs

| DAG | Schedule | Trigger | Description |
|---|---|---|---|
| `weekly_classifier_finetuning` | `@weekly` | Automatic | Full retrain cycle: download data → fine-tune → eval → Slack notification |
| `promote_to_prod` | None | **Manual** | Staging → Production promotion + API rolling restart |
| `batch_analytics` | `@daily` | Automatic | LLM-assisted analytics over classification logs |
| `quality_control` | `@weekly` | Automatic | Model drift detection with configurable F1 threshold |
| `system_maintenance` | Configurable | Automatic | Infrastructure housekeeping |

All training/eval tasks run as `KubernetesPodOperator` pods in the `ml-pipelines` namespace. GPU resources, PVC mounts, and secret injection are declared in Airflow Variables (`deploy/airflow/variables.json`).

The retraining DAG task graph:

```
run_lora_finetuning  ──►  evaluate_staging_model  ──►  request_manual_approval (Slack)
```

Human approval is enforced by design — `promote_to_prod` has `schedule=None` and must be triggered explicitly from the Airflow UI or CLI.

---

## Kubernetes Deployment

**Namespace:** `ml-pipelines`

**Manifests in `deploy/k8s/`:**

| File | Resource |
|---|---|
| `api-deployment.yaml` | Deployment (1 replica, readiness + liveness probes) |
| `api-service.yaml` | ClusterIP Service on port 8000 |
| `api-ingress.yaml` | Ingress with TLS |
| `pvcs.yaml` | PVCs: `pvc-models`, `pvc-data`, `logs-pvc` |
| `configmap.yaml` | Non-secret env vars |
| `secrets.yaml` | API keys, Kaggle credentials, TG bot token |
| `rbac.yaml` | ServiceAccount + ClusterRoleBinding for Airflow workers |
| `monitoring/` | kube-state-metrics + node-exporter DaemonSet |

**Resource requests (API pod):**

| | Requests | Limits |
|---|---|---|
| CPU | 0.5 | 1 |
| Memory | 2 Gi | 4 Gi |

**Training pod (Airflow KubernetesPodOperator):**

| | Requests | Limits |
|---|---|---|
| CPU | 2 | 4 |
| Memory | 8 Gi | 16 Gi |
| GPU | — | 1× `nvidia.com/gpu` |

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request to `main`:

```
Checkout  ──►  Install uv + Python 3.10  ──►  Install deps (api + dev extras)
──►  ruff format --check  ──►  ruff check  ──►  pytest tests/ -v
```

Package management uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible installs.

---

## Demo

A minimal Streamlit UI is provided in `demo/` for quick local evaluation without any infrastructure setup.

```bash
cd demo
pip install -r requirements.txt

# Point at a running API instance
export API_URL=http://localhost:8000/api/v1/classify
export API_KEY=your_key

streamlit run app.py
```

Or run in Docker:

```bash
docker build -t fake-news-demo ./demo
docker run -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000/api/v1/classify \
  fake-news-demo
```

---

## Quickstart (Local)

**Prerequisites:** Python 3.10+, `uv`, Kaggle API credentials.

```bash
# 1. Clone and install
git clone <repo>
cd fake-news-detection-ml-system
uv pip install -e ".[train,api,dev]"

# 2. Configure secrets
cp .env.example .env
# Fill in: KAGGLE_USERNAME, KAGGLE_KEY, MLFLOW_TRACKING_URI, API_KEY, TG_BOT_TOKEN

# 3. Run a quick training experiment (CPU, 50 batches)
python -m src.train trainer.limit_train_batches=50 trainer.limit_val_batches=20

# 4. Start the API
python -m src.run_api

# 5. Test the endpoint
curl -X POST http://localhost:8000/api/v1/classify \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT: You have won a $1000 gift card! Claim now!"}'

# 6. (Optional) Start the Telegram bot in polling mode
python -m src.api.tg_bot.bot_local
```

---

## Roadmap

- Evaluate the system on more recent and diverse spam datasets to improve generalization on adversarial examples.
- Benchmark domain-adapted encoder models pre-trained on email corpora.
- Add online learning support for continuous model updates from production traffic.
- Helm chart for one-command cluster deployment.
