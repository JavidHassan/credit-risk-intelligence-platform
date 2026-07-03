"""
Model Explainability with SHAP
Global and local feature importance, individual explanations, and bias checks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Install with: pip install shap")


class SHAPExplainer:
    """SHAP-based model explainability for credit risk."""

    def __init__(self, model: Any, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def initialize_explainer(self, X_background: pd.DataFrame, max_samples: int = 100):
        """Initialize SHAP explainer with background data."""
        if not HAS_SHAP:
            logger.error("SHAP not available.")
            return

        bg = X_background.sample(min(max_samples, len(X_background)), random_state=42)
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer")
        except Exception:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, bg
            )
            logger.info("Using KernelExplainer")

    def compute_shap_values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Compute SHAP values for dataset."""
        if self.explainer is None:
            logger.error("Explainer not initialized.")
            return None

        self.shap_values = self.explainer.shap_values(X)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]

        logger.info(f"Computed SHAP values for {X.shape[0]} samples")
        return self.shap_values

    def global_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute global SHAP feature importance."""
        if self.shap_values is None:
            self.compute_shap_values(X)

        if self.shap_values is None:
            return pd.DataFrame()

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.abs(self.shap_values).mean(axis=0),
            "std_shap": np.abs(self.shap_values).std(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        logger.info(f"Top 10 features:\n{importance.head(10).to_string(index=False)}")
        return importance

    def explain_customer(self, X_customer: pd.DataFrame, customer_id: str = "") -> Dict:
        """Generate explanation for a single customer prediction."""
        if self.explainer is None:
            logger.error("Explainer not initialized.")
            return {}

        sv = self.explainer.shap_values(X_customer)
        if isinstance(sv, list):
            sv = sv[1]

        values = sv[0] if sv.ndim > 1 else sv
        prob = self.model.predict_proba(X_customer)[:, 1][0]

        feature_contributions = sorted(
            zip(self.feature_names, values, X_customer.iloc[0].values),
            key=lambda x: abs(x[1]), reverse=True
        )

        top_positive = [(f, round(v, 4), round(fv, 4))
                        for f, v, fv in feature_contributions if v > 0][:5]
        top_negative = [(f, round(v, 4), round(fv, 4))
                        for f, v, fv in feature_contributions if v < 0][:5]

        explanation = {
            "customer_id": customer_id,
            "default_probability": round(float(prob), 4),
            "risk_increasing_factors": [
                {"feature": f, "shap_value": v, "feature_value": fv}
                for f, v, fv in top_positive
            ],
            "risk_decreasing_factors": [
                {"feature": f, "shap_value": v, "feature_value": fv}
                for f, v, fv in top_negative
            ],
        }
        return explanation

    def bias_check(self, X: pd.DataFrame, sensitive_features: Dict[str, pd.Series]) -> Dict:
        """Check for model bias across sensitive groups."""
        if self.shap_values is None:
            self.compute_shap_values(X)

        bias_results = {}
        predictions = self.model.predict_proba(X)[:, 1]

        for feature_name, feature_values in sensitive_features.items():
            groups = feature_values.unique()
            group_stats = {}

            for group in groups:
                mask = feature_values == group
                group_stats[str(group)] = {
                    "count": int(mask.sum()),
                    "avg_predicted_pd": round(float(predictions[mask].mean()), 4),
                    "avg_abs_shap": round(
                        float(np.abs(self.shap_values[mask]).mean()), 4
                    ) if self.shap_values is not None else None,
                }

            bias_results[feature_name] = group_stats
            pds = [s["avg_predicted_pd"] for s in group_stats.values()]
            disparity = max(pds) - min(pds)
            logger.info(f"Bias check '{feature_name}': max disparity = {disparity:.4f}")

        return bias_results
