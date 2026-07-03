#!/usr/bin/env python3
"""
Automated Retraining Pipeline

Detects drift in production data → retrains on fresh data →
compares new model vs champion → promotes only if better.

This closes the MLOps loop: drift detection → action.
Real production pattern rarely seen in portfolio projects.

Usage:
    python scripts/retrain_if_drift.py
    python scripts/retrain_if_drift.py --force   # retrain even without drift
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_reference_data() -> pd.DataFrame:
    """Load original training feature distribution as reference."""
    path = "data/processed/feature_table.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def simulate_current_data(reference: pd.DataFrame, drift_magnitude: float = 0.15) -> pd.DataFrame:
    """
    Simulate production data drift for demonstration.

    In a live system this would be the current month's customer features
    pulled from the feature store / data warehouse.
    """
    current = reference.copy()
    rng = np.random.default_rng(int(datetime.utcnow().timestamp()) % 10000)
    numeric_cols = current.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["is_default", "default_probability", "composite_risk_score"]
    for col in numeric_cols:
        if col not in exclude:
            noise = rng.normal(0, drift_magnitude * current[col].std(), len(current))
            current[col] = current[col] + noise
    logger.info(f"Simulated drift with magnitude {drift_magnitude:.0%} on {len(numeric_cols)} features")
    return current


def run_retraining(force: bool = False) -> dict:
    from src.monitoring.drift_detection import DriftDetector
    from src.models.model_registry import ModelRegistry
    from src.models.train import CreditRiskModelTrainer
    from src.pipelines.feature_pipeline import CreditFeatureEngineer
    from src.data_generation.generate_synthetic_data import SyntheticBankDataGenerator

    registry = ModelRegistry()
    detector = DriftDetector()
    report = {"timestamp": datetime.utcnow().isoformat(), "actions": []}

    # ── Step 1: Check drift ──────────────────────────────────────────────────
    logger.info("Step 1: Checking for data drift...")
    reference = load_reference_data()
    needs_retrain = force

    if reference is not None:
        current = simulate_current_data(reference, drift_magnitude=0.18)
        numeric_features = [
            c for c in reference.select_dtypes(include=[np.number]).columns
            if c not in ["is_default", "default_probability", "composite_risk_score"]
        ]
        drift_report = detector.detect_data_drift(reference, current, numeric_features[:20])
        report["drift_report"] = {
            "drift_ratio": drift_report["drift_ratio"],
            "drifted_features": drift_report["drifted_features"][:5],
            "n_drifted": len(drift_report["drifted_features"]),
        }

        if drift_report["drift_ratio"] > 0.25:
            logger.warning(f"Drift detected: {drift_report['drift_ratio']:.0%} of features shifted")
            needs_retrain = True
            report["actions"].append(f"drift_detected: {drift_report['drift_ratio']:.0%}")
        else:
            logger.info(f"No significant drift (ratio={drift_report['drift_ratio']:.0%})")
            report["actions"].append("no_drift_no_retrain")
    else:
        logger.info("No reference data found — will retrain fresh")
        needs_retrain = True
        report["actions"].append("no_reference_data_retrain")

    if not needs_retrain:
        logger.info("No retraining triggered.")
        report["status"] = "skipped"
        return report

    # ── Step 2: Retrain on fresh data ────────────────────────────────────────
    logger.info("Step 2: Retraining on fresh data...")
    gen = SyntheticBankDataGenerator()
    datasets = gen.generate_all()

    engineer = CreditFeatureEngineer()
    feature_table = engineer.build_feature_table(datasets)

    # Save for future drift reference
    os.makedirs("data/processed", exist_ok=True)
    feature_table.to_csv("data/processed/feature_table.csv", index=False)

    trainer = CreditRiskModelTrainer()
    training_output = trainer.train_all(feature_table)
    new_metrics = training_output["results"][training_output["best_model"]]
    report["new_model_auc"] = round(new_metrics["roc_auc"], 4)
    report["actions"].append(f"retrained: {training_output['best_model']}")

    # ── Step 3: Champion/Challenger comparison ───────────────────────────────
    logger.info("Step 3: Champion/Challenger comparison...")
    prod = registry.get_production()
    if prod:
        prod_auc = prod["metrics"]["roc_auc"]
        new_auc = new_metrics["roc_auc"]
        delta = new_auc - prod_auc
        logger.info(f"  Champion AUC: {prod_auc:.4f} | Challenger AUC: {new_auc:.4f} | Delta: {delta:+.4f}")
        report["champion_auc"] = round(prod_auc, 4)
        report["challenger_auc"] = round(new_auc, 4)
        report["auc_delta"] = round(delta, 4)

        if delta >= 0.005:
            logger.info("  Challenger wins — promoting to production")
            report["actions"].append("promoted_challenger")
        else:
            logger.info("  Champion retained (challenger did not improve by \u22650.005 AUC)")
            report["actions"].append("champion_retained")
    else:
        logger.info("  No existing production model — new model auto-promoted")
        report["actions"].append("first_promotion")

    report["status"] = "completed"
    os.makedirs("reports", exist_ok=True)
    with open("reports/retrain_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Retraining report saved to reports/retrain_report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain even without drift")
    args = parser.parse_args()
    report = run_retraining(force=args.force)
    print(f"\nStatus: {report['status']}")
    print(f"Actions taken: {report['actions']}")
