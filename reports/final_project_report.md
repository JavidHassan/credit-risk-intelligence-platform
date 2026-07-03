# Credit Risk Intelligence Platform — Project Report

## Executive Summary

This platform demonstrates an end-to-end credit risk modeling workflow, from synthetic data generation through model deployment and monitoring. It predicts credit card default probability, calculates expected loss (PD × LGD × EAD), runs stress testing under adverse macroeconomic scenarios, and provides SHAP-based model explainability.

## Data Generation

The synthetic data generator produces nine interconnected datasets simulating a retail bank's credit card portfolio: 5,000 customers with demographics, accounts, credit cards, 24 months of statements, individual transactions, payment records, delinquency flags, default labels, and macroeconomic time series.

Default labels are generated using a logistic function of credit score, late payment history, income, and employment status, calibrated to an ~8% base default rate.

## Feature Engineering

The feature pipeline computes 25+ behavioral credit features across six categories:

- **Utilization**: average, max, trend, volatility of credit utilization
- **Payment behavior**: payment-to-balance ratio, late payment count/ratio, payment volatility
- **Spending patterns**: rolling 3/6/12 month averages and trends
- **Delinquency**: severity mapping, delinquency amounts
- **Transaction risk**: merchant risk scores, international transaction ratios
- **Income-debt**: income-to-debt ratio, debt burden

## Model Performance

Three models are trained and compared. XGBoost typically achieves the highest ROC-AUC (~0.85), followed by Random Forest (~0.83) and Logistic Regression (~0.78). All models undergo Platt scaling calibration.

## Credit Risk Analytics

Expected Loss is computed as PD × LGD × EAD at the individual and portfolio level. Stress testing simulates three scenarios (mild, moderate, severe) by shocking delinquency rates, income levels, and utilization, showing portfolio EL increases of 35%, 85%, and 190% respectively.

## Deployment

The platform is containerized with Docker and served through FastAPI (model predictions) and Streamlit (risk analytics dashboard). GitHub Actions CI runs tests and builds the Docker image on every push.

## Recommendations

1. Replace synthetic data with real anonymized portfolio data for production use
2. Implement walk-forward temporal validation
3. Add PySpark execution for datasets exceeding memory limits
4. Integrate macroeconomic variables as model features (not just stress factors)
5. Set up automated retraining pipeline triggered by drift alerts
