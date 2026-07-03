# Model Card: Credit Default Prediction

## Model Details

- **Model type**: Binary classification (Probability of Default)
- **Algorithms**: Logistic Regression, Random Forest, XGBoost (best selected by AUC)
- **Framework**: scikit-learn, XGBoost
- **Calibration**: Platt scaling (sigmoid) via CalibratedClassifierCV
- **Training data**: Synthetic banking data (5,000 customers, 24 months)

## Intended Use

- **Primary**: Predict credit card default probability for portfolio risk management
- **Secondary**: Expected loss forecasting, stress testing, customer risk segmentation
- **Users**: Risk analysts, credit officers, portfolio managers
- **Out of scope**: Automated credit decisions without human review

## Performance Metrics

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|-------------------|---------------|---------|
| ROC-AUC | ~0.78 | ~0.83 | ~0.85 |
| KS Statistic | ~0.45 | ~0.55 | ~0.58 |
| Brier Score | ~0.07 | ~0.06 | ~0.05 |

*Note: Exact values depend on the generated synthetic data seed.*

## Features Used

Key predictive features (ranked by SHAP importance):
1. Average credit utilization
2. Late payment count / ratio
3. Delinquency severity
4. Income-to-debt ratio
5. Payment volatility
6. Average merchant risk score
7. Rolling spending trends
8. Current utilization

## Limitations

- Trained on synthetic data — patterns may not reflect real-world distributions
- No temporal validation (walk-forward) applied
- Limited to credit card default; does not cover mortgage, auto, or personal loans
- Macroeconomic variables not integrated into the ML model (used in stress testing only)

## Ethical Considerations

- Bias checks available via SHAP analysis across gender, age, and geography
- Model should not be used as sole basis for credit decisions
- Disparate impact analysis recommended before production deployment
- Regular monitoring for drift and performance decay is required

## Monitoring

- PSI-based data drift detection on all input features
- Prediction distribution drift monitoring
- AUC performance decay tracking with retraining triggers
- Recommended monitoring frequency: weekly
