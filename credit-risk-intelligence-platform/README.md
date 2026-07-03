# 🏦 Credit Risk Intelligence Platform

A production-style machine learning platform for credit card default prediction, expected loss forecasting, stress testing, model explainability, and drift monitoring.

Built with Python · PySpark · SQL · scikit-learn · XGBoost · SHAP · FastAPI · Streamlit · Docker · GitHub Actions

---

## Overview

This platform simulates how a financial institution predicts credit card default risk and monitors model health over time. It generates realistic synthetic banking data, processes it through scalable pipelines, engineers behavioral credit features, trains probability-of-default models, calculates expected loss, runs macroeconomic stress tests, explains predictions with SHAP, detects data/model drift, serves predictions through an API, and displays risk analytics in an interactive dashboard.

## Architecture

```
Synthetic Data ─► Preprocessing ─► Feature Engineering ─► Model Training
                                                              │
                    ┌─────────────────────────────────────────┤
                    │              │              │            │
              Expected Loss   Stress Test   Explainability   Drift Monitor
                    │              │              │            │
                    └──────────────┴──────────────┴────────────┘
                                        │
                              ┌─────────┴─────────┐
                          FastAPI REST API    Streamlit Dashboard
```

## Features

### Data Generation
Synthetic generator producing 9 interconnected datasets: customers, accounts, credit cards, monthly statements, transactions, payments, delinquencies, defaults, and macroeconomic variables.

### Feature Engineering
25+ behavioral credit features across six categories: credit utilization metrics, payment behavior, rolling spending trends, delinquency history, transaction/merchant risk, and income-to-debt ratios. Customer risk segmentation via composite scoring.

### Machine Learning
Three model comparison (Logistic Regression, Random Forest, XGBoost) with probability calibration, cross-validation, and comprehensive evaluation: ROC-AUC, Precision/Recall, KS statistic, lift charts, Brier score, and confusion matrices.

### Credit Risk Math
- **PD** — Probability of Default (model output)
- **LGD** — Loss Given Default (Beta distribution, ~45% mean)
- **EAD** — Exposure at Default (balance + utilization factor × unused limit)
- **Expected Loss** = PD × LGD × EAD

### Stress Testing
Three scenarios (mild, moderate, severe) shocking unemployment, income, and delinquency rates to project portfolio losses under adverse conditions.

### Explainability
SHAP-based global and local feature importance, individual customer risk explanations, and bias/fairness checks across sensitive attributes.

### Monitoring
PSI-based data drift detection, prediction distribution drift, model performance decay tracking, and automated retraining triggers.

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Single customer default prediction |
| `/batch_predict` | POST | Batch predictions |
| `/model_metrics` | GET | Model performance metrics |
| `/customer_explanation/{id}` | GET | SHAP-based risk explanation |

### Dashboard
Six-page Streamlit dashboard: Portfolio Overview, Risk Segmentation, Expected Loss Analysis, Feature Importance, Stress Testing Results, and Drift Monitoring with alerts.

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/JavidHassan/credit-risk-intelligence-platform.git
cd credit-risk-intelligence-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python -m src.data_generation.generate_synthetic_data

# Run tests
pytest tests/ -v

# Start the API
uvicorn src.api.main:app --reload

# Start the dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

### Docker

```bash
docker-compose up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

## Project Structure

```
credit-risk-intelligence-platform/
├── .github/workflows/ci.yml          # GitHub Actions CI pipeline
├── configs/config.yaml               # Central configuration
├── data/                             # Generated data (gitignored)
├── src/
│   ├── data_generation/              # Synthetic bank data generator
│   ├── pipelines/                    # Preprocessing & feature engineering
│   ├── models/                       # Training, evaluation, prediction
│   ├── risk/                         # Expected loss & stress testing
│   ├── explainability/               # SHAP analysis & bias checks
│   ├── monitoring/                   # Drift detection & retraining triggers
│   ├── api/                          # FastAPI application
│   └── dashboard/                    # Streamlit dashboard
├── sql/                              # Portfolio SQL queries
├── tests/                            # Unit tests
├── docs/                             # Architecture, model card, governance
├── reports/                          # Project report
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| ML | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Data Pipeline | PySpark, Pandas, NumPy |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Infrastructure | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest |

## Documentation

- [Architecture](docs/architecture.md) — System design and data flow
- [Model Card](docs/model_card.md) — Model details, performance, limitations
- [Governance](docs/governance.md) — Model risk governance framework
- [Project Report](reports/final_project_report.md) — Full project summary

## License

MIT
