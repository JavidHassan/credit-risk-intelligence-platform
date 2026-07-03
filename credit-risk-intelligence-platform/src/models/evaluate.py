"""
Model Evaluation Module
ROC-AUC, Precision/Recall, Confusion Matrix, KS Statistic, Lift Chart.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, average_precision_score,
    brier_score_loss,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for credit risk."""

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                 model_name: str) -> Dict:
        """Run full evaluation suite on a model."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = self._compute_metrics(y_test, y_pred, y_prob)
        ks_stat = self._ks_statistic(y_test, y_prob)
        lift = self._lift_chart_data(y_test, y_prob)

        metrics["ks_statistic"] = ks_stat
        metrics["lift_data"] = lift

        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {model_name}")
        logger.info(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        logger.info(f"PR-AUC:      {metrics['pr_auc']:.4f}")
        logger.info(f"Precision:   {metrics['precision']:.4f}")
        logger.info(f"Recall:      {metrics['recall']:.4f}")
        logger.info(f"F1:          {metrics['f1']:.4f}")
        logger.info(f"KS Stat:     {metrics['ks_statistic']:.4f}")
        logger.info(f"Brier Score: {metrics['brier_score']:.4f}")
        logger.info(f"{'='*50}")

        return metrics

    def _compute_metrics(self, y_true, y_pred, y_prob) -> Dict:
        """Compute standard classification metrics."""
        cm = confusion_matrix(y_true, y_pred)
        return {
            "roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "brier_score": brier_score_loss(y_true, y_prob),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
        }

    def _ks_statistic(self, y_true, y_prob) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks = np.max(tpr - fpr)
        return round(ks, 4)

    def _lift_chart_data(self, y_true, y_prob, n_bins: int = 10) -> Dict:
        """Compute lift chart data by decile."""
        df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
        df["decile"] = pd.qcut(df["y_prob"], n_bins, labels=False, duplicates="drop")

        lift_data = df.groupby("decile").agg(
            count=("y_true", "size"),
            events=("y_true", "sum"),
            avg_prob=("y_prob", "mean"),
        ).reset_index()

        overall_rate = y_true.mean()
        lift_data["event_rate"] = lift_data["events"] / lift_data["count"]
        lift_data["lift"] = lift_data["event_rate"] / overall_rate
        lift_data["cumulative_events"] = lift_data["events"].cumsum()
        lift_data["cumulative_event_rate"] = (
            lift_data["cumulative_events"] / lift_data["events"].sum()
        )

        return lift_data.to_dict(orient="records")

    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Side-by-side comparison of all trained models."""
        comparison = []
        for name, metrics in results.items():
            comparison.append({
                "model": name,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "ks_statistic": metrics["ks_statistic"],
                "brier_score": metrics["brier_score"],
            })
        df = pd.DataFrame(comparison).sort_values("roc_auc", ascending=False)
        logger.info(f"\nModel Comparison:\n{df.to_string(index=False)}")
        return df
