# Model Risk Governance

## Overview

This document outlines the governance framework for the Credit Risk Intelligence Platform, covering model development, validation, deployment, and ongoing monitoring.

## Model Development Standards

### Data Quality
- All input data must pass validation checks (missing values, range checks, duplicates)
- Data lineage is tracked from raw generation through feature engineering
- Feature distributions are logged at training time for drift comparison

### Model Selection
- Minimum three candidate algorithms evaluated per training cycle
- Model selection based on ROC-AUC with secondary consideration for calibration (Brier score)
- All models undergo probability calibration before deployment

### Documentation
- Model card maintained with performance metrics, limitations, and ethical considerations
- Architecture documentation kept current with system changes
- All configuration managed through version-controlled YAML files

## Validation

### Pre-Deployment
- Unit tests cover feature engineering, expected loss calculations, and model evaluation
- Cross-validation (5-fold) required for all candidate models
- Lift chart and KS statistic analysis for discrimination power

### Ongoing
- Weekly drift monitoring (PSI on features and predictions)
- Monthly performance benchmarking against holdout data
- Quarterly full model review

## Retraining Policy

Retraining is triggered when any of these conditions are met:
- Prediction PSI exceeds 0.25
- ROC-AUC decays by more than 5% from baseline
- More than 30% of input features show significant drift (PSI > 0.2)
- Quarterly scheduled retraining regardless of drift metrics

## Roles and Responsibilities

- **Model Developer**: Build, test, and document models
- **Model Validator**: Independent review of model performance and assumptions
- **Risk Officer**: Approve model for production use
- **MLOps Engineer**: Deploy, monitor, and manage retraining pipelines

## Audit Trail

- All model artifacts versioned and stored with training metadata
- Git history provides full code change audit trail
- Monitoring reports archived for regulatory review
