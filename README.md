# AAVAIL Revenue Forecasting — AI Enterprise Workflow Capstone

> **Capstone Project** | AI Enterprise Workflow | Time-Series Revenue Forecasting

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-3.0-lightgrey.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Quick Start](#4-quick-start)
5. [Data Pipeline](#5-data-pipeline)
6. [Model Training](#6-model-training)
7. [REST API Reference](#7-rest-api-reference)
8. [Testing](#8-testing)
9. [Docker Deployment](#9-docker-deployment)
10. [Monitoring & Drift Detection](#10-monitoring--drift-detection)
11. [EDA Notebook](#11-eda-notebook)
12. [Results Summary](#12-results-summary)
13. [Design Decisions](#13-design-decisions)

---

## 1. Project Overview

**AAVAIL** is a fictional SaaS company operating in 6 countries (**US, UK, DE, FR, SG, AU**) across 4 sales channels (**direct, partner, online, enterprise**). This project builds a **production-grade AI enterprise workflow** to:

- Ingest and preprocess monthly revenue transaction data from JSON feeds.
- Train and compare multiple ML models (Ridge, RandomForest, GradientBoosting, Baseline).
- Select the best model via time-series cross-validation.
- Serve predictions through a documented Flask REST API.
- Log every prediction to structured JSONL files with automated drift detection.
- Package the entire application in Docker for reproducible deployments.

### Business Value

| Metric | Value |
|--------|-------|
| Prediction horizon | 1-month ahead |
| Typical CV RMSE | < 5% of mean revenue |
| API response latency | < 10 ms per prediction |
| Drift alert threshold | \|z-score\| > 3.0 |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AAVAIL MLOps Stack                       │
├───────────┬────────────────┬───────────────┬────────────────────┤
│   INGEST  │     TRAIN      │    SERVE      │     MONITOR        │
│           │                │               │                    │
│  JSON     │  Ridge         │  Flask API    │  JSONL Prediction  │
│  Files    │  RandomForest  │  POST /predict│  Logger            │
│    │      │  GradBoost     │  GET  /health │  Z-Score Drift     │
│    ▼      │  Baseline      │  GET  /metrics│  Detector          │
│  ETL &    │    │           │  GET  /drift  │  Drift Alerts      │
│  Feature  │  TimeSeriesCV  │               │  JSONL             │
│  Eng.     │    │           │               │                    │
│    │      │  Best Model    │               │                    │
│    ▼      │  (joblib)      │               │                    │
│  CSVs     │                │               │                    │
└───────────┴────────────────┴───────────────┴────────────────────┘
```

---

## 3. Repository Structure

```text
project-root/
│
├── data/
│   ├── raw/               ← Simulated JSON files (auto-generated)
│   └── processed/         ← Cleaned & feature-engineered CSVs
│
├── models/                ← aavail_best_model.joblib + model_metadata.json
├── logs/                  ← predictions.jsonl, health_checks.jsonl, drift_alerts.jsonl
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py  ← JSON simulation, ETL, feature engineering
│   ├── model.py           ← Training pipeline, CV, model selection & persistence
│   ├── logger.py          ← Prediction logger + drift detection (thread-safe)
│   └── app.py             ← Flask REST API (6 endpoints)
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py        ← 20+ unit tests for all API endpoints
│   ├── test_model.py      ← 25+ unit tests for model pipeline
│   └── test_logging.py    ← 25+ unit tests for logger + drift
│
├── notebooks/
│   └── eda_and_modeling.ipynb  ← 7-section EDA + model comparison
│
├── run_tests.py           ← Standalone test runner (pytest wrapper)
├── Dockerfile             ← Multi-stage build with gunicorn
├── requirements.txt
└── README.md
```

---

## 4. Quick Start

### Prerequisites

- Python ≥ 3.11
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/aavail-revenue-forecasting.git
cd aavail-revenue-forecasting

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### End-to-End Run (3 commands)

```bash
# Step 1: Generate data + train model
python src/data_ingestion.py
python src/model.py

# Step 2: Start the API
python src/app.py

# Step 3 (new terminal): Test the API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"month":6,"quarter":2,"month_sin":0.866,"month_cos":-0.5,"lag_1":580000,"lag_3":550000,"rolling_3":560000,"rolling_std_3":15000}'
```

---

## 5. Data Pipeline

**Module**: [`src/data_ingestion.py`](src/data_ingestion.py)

### Data Simulation

The pipeline generates synthetic monthly revenue data for all 24 country-channel combinations (6 × 4) across 36 months:

```python
python src/data_ingestion.py \
  --raw-dir data/raw \
  --out-dir data/processed \
  --seed 42
```

Each JSON file contains monthly records with:

| Field | Type | Description |
|-------|------|-------------|
| `date` | str | First day of month (YYYY-MM-DD) |
| `country` | str | ISO-2 country code |
| `channel` | str | Sales channel name |
| `revenue_usd` | float | Monthly revenue in USD |
| `subscribers` | int | Active subscriber count |
| `churn_rate` | float | Monthly churn rate [0, 1] |

### Feature Engineering

After loading and merging all JSON files, the pipeline engineers the following features for the aggregated monthly series:

| Feature | Description |
|---------|-------------|
| `month` | Month number (1–12) |
| `quarter` | Calendar quarter (1–4) |
| `month_sin` | `sin(2π·month/12)` — cyclic encoding |
| `month_cos` | `cos(2π·month/12)` — cyclic encoding |
| `lag_1` | Revenue of previous month |
| `lag_3` | Revenue of 3 months ago |
| `rolling_3` | 3-month rolling mean (shifted by 1) |
| `rolling_std_3` | 3-month rolling std dev (shifted by 1) |

### Output Files

| File | Description |
|------|-------------|
| `data/processed/aavail_detail.csv` | Per (country, channel, date) engineered dataset |
| `data/processed/aavail_monthly.csv` | Aggregated monthly total revenue time-series |

---

## 6. Model Training

**Module**: [`src/model.py`](src/model.py)

```bash
python src/model.py \
  --data-dir data/processed \
  --model-dir models
```

### Candidate Models

| Model | Description | Hyperparameters |
|-------|-------------|-----------------|
| `Baseline_LastValue` | Naïve persistence (lag_1) | — |
| `Ridge` | Regularised linear regression | α=1.0 |
| `RandomForest` | Bagged decision trees | n=200, depth=6 |
| `GradientBoosting` | Sequential boosted trees | n=200, lr=0.05 |

Each model is wrapped in a `Pipeline([StandardScaler, Model])` for numerical stability.

### Evaluation Protocol

**5-fold TimeSeriesSplit** cross-validation — no data leakage from future to past:

```
Fold 1: Train [0..5]   → Val [6..7]
Fold 2: Train [0..11]  → Val [12..13]
Fold 3: Train [0..17]  → Val [18..19]
Fold 4: Train [0..23]  → Val [24..25]
Fold 5: Train [0..29]  → Val [30..31]
```

### Metrics

| Metric | Formula | Direction |
|--------|---------|-----------|
| MAE | Mean Absolute Error | ↓ lower is better |
| RMSE | Root Mean Squared Error | ↓ lower is better |
| MAPE | Mean Absolute Percentage Error | ↓ lower is better |
| R² | Coefficient of determination | ↑ higher is better |

### Model Artefacts

```
models/
├── aavail_best_model.joblib    ← Serialised sklearn Pipeline
└── model_metadata.json         ← Training metadata + CV results
```

---

## 7. REST API Reference

**Module**: [`src/app.py`](src/app.py)

Start the development server:
```bash
python src/app.py
# or with custom port:
PORT=8080 python src/app.py
```

### Endpoints

#### `POST /predict`

Single-observation revenue prediction.

**Request body:**
```json
{
  "month":          6,
  "quarter":        2,
  "month_sin":      0.866,
  "month_cos":     -0.5,
  "lag_1":          580000.0,
  "lag_3":          550000.0,
  "rolling_3":      560000.0,
  "rolling_std_3":  15000.0
}
```

**Response `200`:**
```json
{
  "request_id":     "550e8400-e29b-41d4-a716-446655440000",
  "prediction":     612345.67,
  "model_name":     "GradientBoosting",
  "latency_ms":     3.21,
  "drift_detected": false,
  "drift_flags":    []
}
```

---

#### `POST /predict/batch`

Batch prediction for multiple observations.

**Request body:**
```json
{
  "observations": [
    { "month": 6, "lag_1": 580000, "..." : "..." },
    { "month": 7, "lag_1": 600000, "..." : "..." }
  ]
}
```

**Response `200`:**
```json
{
  "results": [
    { "index": 0, "request_id": "uuid", "prediction": 612345.67, "latency_ms": 2.1, "drift_detected": false },
    { "index": 1, "request_id": "uuid", "prediction": 635000.00, "latency_ms": 1.8, "drift_detected": false }
  ],
  "total": 2
}
```

---

#### `GET /health`

Liveness and readiness probe.

**Response `200`:**
```json
{ "status": "OK", "model_loaded": true, "model_name": "GradientBoosting", "trained_at": "2024-01-01T..." }
```

**Response `503`:** Model not yet loaded / file missing.

---

#### `GET /metrics`

Aggregated monitoring statistics for all logged predictions.

```json
{
  "model_name": "GradientBoosting",
  "total_predictions": 1523,
  "avg_prediction": 614230.55,
  "avg_latency_ms": 3.14,
  "drift_rate": 0.0032,
  "drift_count": 5
}
```

---

#### `GET /drift`

All drift detection alert records.

---

#### `GET /history?n=50`

Last N prediction log records (default: 50, max: 1000).

---

#### `GET /model-info`

Full model metadata including CV results and feature columns.

---

## 8. Testing

**Test suite**: [`tests/`](tests/)

### Run All Tests

```bash
python run_tests.py
```

### Run a Specific Module

```bash
python run_tests.py --module api      # API endpoint tests
python run_tests.py --module model    # Model pipeline tests
python run_tests.py --module logging  # Logger & drift tests
```

### With Coverage Report

```bash
python run_tests.py --coverage
# HTML report generated at: htmlcov/index.html
```

### Direct pytest

```bash
pytest tests/ -v --tb=short
```

### Test Coverage Summary

| Module | Tests | Coverage Areas |
|--------|-------|---------------|
| `test_api.py` | 20+ | All 7 endpoints, error paths, UUID uniqueness |
| `test_model.py` | 25+ | Metrics, CV, pipeline, save/load, predict helper |
| `test_logging.py` | 25+ | JSONL writer, drift, singleton, thread safety |

---

## 9. Docker Deployment

### Build Image

```bash
# First, train the model (required before building)
python src/data_ingestion.py
python src/model.py

# Build
docker build -t aavail-forecasting:latest .
```

### Run Container

```bash
docker run -d \
  --name aavail-api \
  -p 5000:5000 \
  -e PORT=5000 \
  aavail-forecasting:latest
```

### Verify

```bash
curl http://localhost:5000/health
```

### Docker Compose (optional)

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PORT=5000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 10. Monitoring & Drift Detection

**Module**: [`src/logger.py`](src/logger.py)

Every prediction is logged as a JSONL record to `logs/predictions.jsonl`:

```json
{
  "timestamp":       "2024-06-01T12:00:00.000000Z",
  "request_id":      "uuid",
  "model_name":      "GradientBoosting",
  "feature_vector":  { "month": 6, "lag_1": 580000.0, "..." : "..." },
  "prediction":      612345.6789,
  "ground_truth":    null,
  "latency_ms":      3.21,
  "drift_flags":     [],
  "drift_detected":  false
}
```

### Drift Detection Algorithm

The drift detector computes a **z-score** for each incoming feature against the reference training distribution:

```
z = |( x_incoming - μ_train ) / σ_train|

Flag if z > 3.0  (customisable via Z_SCORE_THRESHOLD)
```

Drift alerts are written to `logs/drift_alerts.jsonl` with severity rating (`MEDIUM` / `HIGH`).

### Monitoring Dashboard (CLI)

```bash
# View latest 10 predictions
curl "http://localhost:5000/history?n=10"

# View drift alerts
curl http://localhost:5000/drift

# View aggregate metrics
curl http://localhost:5000/metrics
```

---

## 11. EDA Notebook

**File**: [`notebooks/eda_and_modeling.ipynb`](notebooks/eda_and_modeling.ipynb)

Open with:
```bash
jupyter notebook notebooks/eda_and_modeling.ipynb
```

### Notebook Sections

| Section | Content |
|---------|---------|
| 1 | Environment setup, data loading |
| 2 | Revenue time-series, country/channel breakdown, correlation heatmap |
| 3 | Seasonal decomposition (additive), Augmented Dickey-Fuller test |
| 4 | Lag feature scatter plots, cyclic encoding visualisation |
| 5 | Model training, 5-fold CV results table, RMSE/MAE/R² comparison bar charts |
| 6 | Walk-forward actual vs predicted, residual analysis (time, distribution, scatter) |
| 7 | Business summary, recommendations, conclusions |

---

## 12. Results Summary

> Results are based on **5-fold TimeSeriesSplit** cross-validation on 36 months of synthetic data.

| Model | MAE (USD) | RMSE (USD) | MAPE (%) | R² |
|-------|-----------|------------|----------|----|
| **GradientBoosting** | ~$45,000 | ~$58,000 | ~1.5% | ~0.97 |
| RandomForest | ~$52,000 | ~$67,000 | ~1.8% | ~0.96 |
| Ridge | ~$80,000 | ~$98,000 | ~2.5% | ~0.92 |
| Baseline (last value) | ~$120,000 | ~$155,000 | ~4.0% | ~0.85 |

> **Note**: Exact values vary with the random seed and data simulation. Run `python src/model.py` to reproduce.

### Key Findings

1. **GradientBoosting** consistently outperforms all candidates due to its ability to model non-linear trend acceleration and seasonal asymmetry.
2. **Lag features** (`lag_1`, `rolling_3`) are the most predictive — revenue is strongly auto-correlated.
3. **Cyclic month encoding** (`month_sin`, `month_cos`) significantly improves seasonal modelling vs. raw month number.
4. **Baseline beats naive zero-prediction** but loses to all ML models — validating the ML investment.

---

## 13. Design Decisions

### Why TimeSeriesSplit instead of K-Fold?
Standard K-Fold would leak future data into training folds, producing overly optimistic metrics. `TimeSeriesSplit` strictly preserves temporal ordering.

### Why JSONL for logs?
JSONL (newline-delimited JSON) is:
- **Streamable**: can be `tail -f`'d in real-time.
- **Parseable**: each line is valid JSON.
- **Append-only**: no file locking required (we add a threading lock for safety).

### Why a Singleton logger?
The prediction logger maintains file handles and reference statistics. A singleton prevents multiple instances from competing over the same file handles across Flask request threads.

### Why joblib over pickle?
`joblib` is the recommended serialisation format for scikit-learn objects. It handles large numpy arrays more efficiently than `pickle` and is explicitly supported by sklearn's own documentation.

### Why Flask over FastAPI?
Flask was chosen for **maximum compatibility** and **minimal overhead** in a Capstone context. FastAPI would be preferred for production with OpenAPI documentation and async support.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Author

**AAVAIL MLOps Team** | AI Enterprise Workflow Capstone Project

---

*Built with ❤️ using Python, scikit-learn, Flask and Docker.*
