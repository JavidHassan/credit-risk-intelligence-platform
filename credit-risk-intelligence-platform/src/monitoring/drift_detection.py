"""
Data and Model Drift Detection
PSI-based drift monitoring, feature distribution checks, and retraining triggers.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Monitors data and model drift for production models."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.config = cfg["monitoring"]
        self.drift_threshold = self.config["drift_threshold"]
        self.psi_threshold = self.config["psi_threshold"]
        self.retrain_threshold = self.config["retrain_trigger_psi"]

    @staticmethod
    def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions."""
        eps = 1e-6
        breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
        cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current) + eps

        psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
        return round(float(psi), 6)

    def detect_data_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict:
        """Detect drift in feature distributions."""
        logger.info("Running data drift detection...")
        drift_results = {}
        drifted_features = []

        for col in feature_names:
            if col not in reference.columns or col not in current.columns:
                continue
            if reference[col].dtype not in ["float64", "int64"]:
                continue

            ref = reference[col].dropna().values
            cur = current[col].dropna().values

            if len(ref) == 0 or len(cur) == 0:
                continue

            psi = self.compute_psi(ref, cur)
            mean_shift = abs(cur.mean() - ref.mean()) / (ref.std() + 1e-8)

            is_drifted = psi > self.psi_threshold
            if is_drifted:
                drifted_features.append(col)

            drift_results[col] = {
                "psi": psi,
                "mean_shift": round(float(mean_shift), 4),
                "ref_mean": round(float(ref.mean()), 4),
                "cur_mean": round(float(cur.mean()), 4),
                "ref_std": round(float(ref.std()), 4),
                "cur_std": round(float(cur.std()), 4),
                "is_drifted": is_drifted,
            }

        logger.info(f"Drifted features ({len(drifted_features)}/{len(drift_results)}): {drifted_features}")
        return {
            "feature_drift": drift_results,
            "drifted_features": drifted_features,
            "drift_ratio": round(len(drifted_features) / max(len(drift_results), 1), 4),
        }

    def detect_prediction_drift(
        self, reference_preds: np.ndarray, current_preds: np.ndarray
    ) -> Dict:
        """Detect drift in model predictions."""
        psi = self.compute_psi(reference_preds, current_preds)
        mean_shift = abs(current_preds.mean() - reference_preds.mean())

        result = {
            "prediction_psi": psi,
            "mean_shift": round(float(mean_shift), 4),
            "ref_mean_pred": round(float(reference_preds.mean()), 4),
            "cur_mean_pred": round(float(current_preds.mean()), 4),
            "is_drifted": psi > self.psi_threshold,
        }

        logger.info(f"Prediction drift PSI: {psi:.4f} "
                     f"({'DRIFTED' if result['is_drifted'] else 'OK'})")
        return result

    def check_performance_decay(
        self, baseline_auc: float, current_auc: float
    ) -> Dict:
        """Check if model performance has decayed significantly."""
        decay = baseline_auc - current_auc
        decay_pct = (decay / baseline_auc) * 100

        result = {
            "baseline_auc": round(baseline_auc, 4),
            "current_auc": round(current_auc, 4),
            "auc_decay": round(decay, 4),
            "decay_pct": round(decay_pct, 2),
            "needs_retraining": decay > self.config["performance_decay_threshold"],
        }

        status = "RETRAIN NEEDED" if result["needs_retraining"] else "OK"
        logger.info(f"Performance decay: {decay_pct:.1f}% ({status})")
        return result

    def generate_monitoring_report(
        self, data_drift: Dict, prediction_drift: Dict,
        performance: Dict
    ) -> Dict:
        """Generate comprehensive monitoring report."""
        needs_retrain = (
            prediction_drift.get("is_drifted", False)
            or performance.get("needs_retraining", False)
            or data_drift.get("drift_ratio", 0) > 0.3
        )

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_drift_summary": {
                "total_features_checked": len(data_drift.get("feature_drift", {})),
                "drifted_features": data_drift.get("drifted_features", []),
                "drift_ratio": data_drift.get("drift_ratio", 0),
            },
            "prediction_drift": prediction_drift,
            "performance": performance,
            "recommendation": "RETRAIN" if needs_retrain else "MONITOR",
            "needs_retraining": needs_retrain,
        }

        logger.info(f"Monitoring recommendation: {report['recommendation']}")
        return report
