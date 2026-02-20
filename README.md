# Credit Scoring — End-to-End MLOps Pipeline

**Project OC6 · AI Engineer Program OpenClassrooms · Pierre Pluton · 2026**

[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-yellow.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF.svg)](https://github.com/features/actions)
[![MLFlow](https://img.shields.io/badge/MLFlow-Tracking-blue.svg)](https://mlflow.org/)

---

## Use Case

Predict **loan default risk** for a microfinance institution (Home Credit Default Risk).

- **8 relational tables**, 57M+ rows aggregated into a single client-level dataset
- **Heavily imbalanced dataset**: 91.9% good clients vs 8.1% defaults (11.4:1 ratio)
- **Asymmetric business cost**: a missed default costs 10x more than a wrongly rejected good client

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **ML / Data** | LightGBM, scikit-learn, imbalanced-learn, pandas, numpy |
| **Experiment tracking** | MLflow (model registry, tracking UI) |
| **API** | FastAPI, Pydantic v2, Uvicorn |
| **Monitoring** | Streamlit (5-tab dashboard), Evidently AI, scipy (KS test) |
| **Database** | Neon (serverless Postgres) |
| **Logging** | Fluentd (Docker log aggregation) → Grafana |
| **Containerization** | Docker, Docker Compose (4 services) |
| **CI/CD** | GitHub Actions (test → build → deploy) |
| **Deployment** | Render (API + Dashboard) |
| **Deps** | uv (astral-sh) with lockfile |
| **Inference optimization** | ONNX Runtime (CPU), skl2onnx, onnxmltools |

---

## Pipeline Architecture

```
Raw data (8 CSVs, 57M+ rows)
  → Hierarchical aggregation        307k clients × 305 features
  → Preprocessing + Feature Eng.    307k × 419 features, 0 NaN, scaled
  → MLflow modeling                 LightGBM, optimal threshold 0.494
  → Export to artifacts/            model.pkl, scaler.pkl, feature_names.json, model.onnx
  → FastAPI + Docker                /predict → probability + APPROVED/REFUSED
  → CI/CD GitHub Actions            test → build container → deploy to Render
```

---

## Project Structure

```
OC6_MLOPS/
├── api/                    # FastAPI app (routes, predict, schemas, config, database)
├── artifacts/              # Exported model (model.pkl, scaler.pkl, metadata)
├── monitoring/             # Streamlit dashboard + drift detection
├── tests/                  # 19 pytest tests (API, prediction, drift)
├── src/                    # Reusable modules (metrics, data_processing)
├── notebooks/              # 6-notebook pipeline (EDA → Preprocessing → Modeling → Inference Opt.)
├── scripts/                # Model export, demo data generation
├── fluentd/                # Docker log aggregation config
├── grafana/                # Provisioned dashboards + Neon datasource
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile              # Production image
├── docker-compose.yml      # Local stack (API + Dashboard + Fluentd + Grafana)
└── render.yaml             # Cloud deployment config
```

---

## Work Done

### Feature Engineering

- **Hierarchical aggregation** of 8 tables (57M+ rows): `bureau_balance` → `bureau` → client, `previous_application` → POS/CC/Installments → client
- **5 "Has_History" features** created **before** imputation to capture missing history as a business signal
- **Semantic imputation** with 5 strategies: AMT→0 (no credit), CNT→0, DAYS→-999 (sentinel), MEAN→median, others→median
- **11 business features**: CREDIT_INCOME_RATIO, AGE_YEARS, EXT_SOURCE_MEAN/PROD, etc.
- Result: 419 features, 0 NaN, 0 Inf, scaler fit on train only (no data leakage)

### Modeling

- **Asymmetric cost score**: `cost = (FN × 10) + FP` — a missed default costs 10x more
- **5 baselines** compared via MLflow (LogReg, Random Forest, XGBoost, LightGBM)
- **GridSearchCV** on LightGBM + **optimal threshold** 0.494 (vs default 0.5)
- Results: AUC = 0.7852, Business Cost = 0.4907
- **Full MLflow tracking**: params, metrics, confusion matrices, model registry

### Production & Monitoring

- **FastAPI serving**: 3 endpoints (`/health`, `/predict`, `/model-info`), resilient to partial features
- **Dual logging**: local JSONL + Postgres (Neon) — DB logging is fail-safe (never crashes the API)
- **5-tab Streamlit dashboard**: client scoring, score distribution, API latency (P50/P95), data drift (KS test + Evidently AI), model metadata
- **Drift detection**: simulation (gradual/sudden/feature shift) + KS 2-sample test per feature
- **Log pipeline**: Fluentd (Docker log driver) → API → Postgres → Grafana
- **3-step CI/CD**: lint (ruff) + tests → Docker build + health check with retry → deploy to Render
- **Docker Compose**: 4 services (API, Dashboard, Fluentd, Grafana)

### Inference Optimization

- **Profiling** (cProfile + timeit) : la construction du DataFrame pandas est le principal
  goulot d'étranglement (23% de la latence LightGBM par appel), pas `predict_proba` lui-même
- **Conversion ONNX** : modèle LightGBM exporté via onnxmltools, 419 features préservées,
  tolérance numérique 7.3e-08 vs LightGBM natif (< 1e-4)
- **Dual inference path** : l'API détecte `artifacts/model.onnx` au démarrage et bascule
  automatiquement ; fallback LightGBM si absent — zéro breaking change
- `_predict_onnx()` construit un numpy float32 array directement (sans DataFrame pandas)
- **Résultats** (500 runs, inference single-row) :

| Engine        | Mean (ms) | P50 (ms) | P95 (ms) | Artefact  |
|---------------|-----------|----------|----------|-----------|
| LightGBM      | 0.71      | 0.70     | 0.77     | 1.12 MB   |
| ONNX Runtime  | 0.01      | 0.01     | 0.01     | 0.71 MB   |
| **Speedup**   | **57.7x** | **58.1x**| **60.9x**| —         |

- Notebook : `notebooks/06_inference_optimization.ipynb`
- Artefact : `artifacts/model.onnx`

---

## Quickstart

```bash
git clone <repo-url> && cd OC6_MLOPS
uv sync                              # install dependencies
uv run pytest tests/ -v              # run 19 tests
uv run python main.py                # API at http://localhost:8000
uv run streamlit run monitoring/dashboard.py  # Dashboard at http://localhost:8501
```

**With Docker:**

```bash
docker compose up                    # API + Dashboard + Fluentd + Grafana
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API status + model loaded |
| `/predict` | POST | Scoring: probability + APPROVED/REFUSED decision |
| `/model-info` | GET | Model metadata |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"SK_ID_CURR": 100001, "features": {"AMT_CREDIT": 0.5, "AMT_ANNUITY": -0.3}}'
```

```json
{
  "SK_ID_CURR": 100001,
  "probability": 0.38,
  "prediction": 0,
  "threshold": 0.494,
  "decision": "APPROVED",
  "inference_time_ms": 2.5
}
```

---

## MLOps Skills Demonstrated

- **Feature engineering** on complex relational data (multi-level aggregation)
- **Business-driven modeling** (asymmetric cost score, optimized threshold)
- **Experiment tracking** and reproducibility (MLflow, model registry)
- **Production serving API** (FastAPI, Pydantic, fail-safe logging)
- **Production monitoring** (real-time dashboard, drift detection, observability)
- **Containerization** and orchestration (Docker, Docker Compose, 4 services)
- **Automated CI/CD** (GitHub Actions: test → build → deploy)
- **Cloud deployment** (Render, Neon serverless Postgres)
- **Automated testing** (19 pytest tests: API, inference, drift)
- **Inference optimization** (ONNX export, onnxruntime CPU, profiling cProfile/timeit, benchmark P50/P95, 57x speedup)

---

**Pierre Pluton** · pierre.pluton@outlook.fr · pierre@thoughtside.com
