# Architecture

## System Overview

The Credit Risk Intelligence Platform follows a modular pipeline architecture where each component is independently testable and deployable.

## Data Flow

```
Synthetic Data Generator
        │
        ▼
   Raw CSV Files
        │
        ▼
 Preprocessing Pipeline ──── Data Validation
        │
        ▼
 Feature Engineering ──────── PySpark / Pandas
        │
        ▼
  Feature Table (ML-ready)
        │
        ├──► Model Training ──► Model Artifacts (.pkl)
        │         │
        │         ├──► Evaluation (ROC, KS, Lift)
        │         └──► Calibration
        │
        ├──► Expected Loss ──► PD × LGD × EAD
        │
        ├──► Stress Testing ──► Scenario Analysis
        │
        ├──► SHAP Explainability ──► Feature Importance
        │
        └──► Drift Monitoring ──► PSI / Performance Decay
                  │
                  ▼
        Retraining Trigger
```

## API Layer

FastAPI serves the trained model through REST endpoints:

- `POST /predict` — Single customer prediction
- `POST /batch_predict` — Batch predictions
- `GET /model_metrics` — Performance metrics
- `GET /health` — Service health check
- `GET /customer_explanation/{id}` — SHAP explanation

## Dashboard

Streamlit dashboard with six pages:

1. Portfolio Overview — KPIs, distributions, scatter plots
2. Risk Segmentation — Segment breakdowns, geographic analysis
3. Expected Loss — EL distribution, top-risk customers
4. Feature Importance — SHAP-based importance rankings
5. Stress Testing — Scenario comparison charts
6. Drift Monitoring — PSI trends, performance tracking

## Deployment

Docker Compose runs two services:
- `api` on port 8000 (FastAPI + Uvicorn)
- `dashboard` on port 8501 (Streamlit)

CI/CD via GitHub Actions runs tests and builds the Docker image on every push.
