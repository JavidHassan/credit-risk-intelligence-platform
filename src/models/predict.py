"""
Prediction Module
Load trained model and generate predictions for new customers.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """Generates default probability predictions."""

    def __init__(self, model_path: str = "models/best_model.pkl",
                 feature_names_path: str = "models/feature_names.pkl"):
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)
        logger.info(f"Loaded model from {model_path}")

    def predict(self, features: pd.DataFrame) -> Dict:
        """Predict default probability for a single customer."""
        X = self._align_features(features)
        prob = self.model.predict_proba(X)[:, 1][0]
        pred = int(prob >= 0.5)

        return {
            "default_probability": round(float(prob), 4),
            "prediction": pred,
            "risk_level": self._risk_level(prob),
        }

    def predict_batch(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict default probabilities for multiple customers."""
        X = self._align_features(features)
        probs = self.model.predict_proba(X)[:, 1]

        results = features[["customer_id"]].copy() if "customer_id" in features.columns else pd.DataFrame()
        results["default_probability"] = probs.round(4)
        results["prediction"] = (probs >= 0.5).astype(int)
        results["risk_level"] = [self._risk_level(p) for p in probs]
        return results

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Ensure input features match training feature names."""
        X = features.reindex(columns=self.feature_names, fill_value=0)
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        return X

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob < 0.1:
            return "low"
        elif prob < 0.3:
            return "medium"
        elif prob < 0.6:
            return "high"
        return "critical"
