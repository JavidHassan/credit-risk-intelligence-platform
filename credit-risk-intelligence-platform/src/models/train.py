"""
Model Training Module
Trains Logistic Regression, Random Forest, and XGBoost models
for probability-of-default prediction with calibration.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import yaml

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModelTrainer:
    """Trains and compares credit risk models."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.config = cfg["model"]
        self.feature_config = cfg["features"]
        self.test_size = self.config["test_size"]
        self.random_state = self.config["random_state"]
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = ""
        self.scaler = StandardScaler()

        os.makedirs("models", exist_ok=True)

    def prepare_data(self, feature_table: pd.DataFrame) -> Tuple:
        """Split features and target, handle train/test split."""
        target = self.feature_config["target_column"]
        drop_cols = [
            self.feature_config["id_column"], target, "default_probability",
            "risk_segment", "composite_risk_score"
        ]
        drop_cols = [c for c in drop_cols if c in feature_table.columns]

        X = feature_table.drop(columns=drop_cols)
        y = feature_table[target]

        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Default rate — train: {y_train.mean():.2%}, test: {y_test.mean():.2%}")

        self.feature_names = list(X.columns)
        return X_train, X_test, y_train, y_test

    def train_logistic_regression(self, X_train, y_train) -> Pipeline:
        """Train calibrated Logistic Regression."""
        logger.info("Training Logistic Regression...")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=self.random_state, C=0.1
            ))
        ])
        pipe.fit(X_train, y_train)
        cv_score = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
        logger.info(f"Logistic Regression CV AUC: {cv_score:.4f}")
        return pipe

    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Train Random Forest classifier."""
        logger.info("Training Random Forest...")
        params = self.config["random_forest_params"]
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
        logger.info(f"Random Forest CV AUC: {cv_score:.4f}")
        return model

    def train_xgboost(self, X_train, y_train) -> Any:
        """Train XGBoost classifier."""
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed, skipping.")
            return None

        logger.info("Training XGBoost...")
        params = self.config["xgboost_params"]
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="auc",
            use_label_encoder=False,
        )
        model.fit(X_train, y_train)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
        logger.info(f"XGBoost CV AUC: {cv_score:.4f}")
        return model

    def calibrate_model(self, model, X_train, y_train) -> CalibratedClassifierCV:
        """Apply Platt scaling for probability calibration."""
        calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        calibrated.fit(X_train, y_train)
        return calibrated

    def train_all(self, feature_table: pd.DataFrame) -> Dict:
        """Train all models and select the best one."""
        from src.models.evaluate import ModelEvaluator

        X_train, X_test, y_train, y_test = self.prepare_data(feature_table)
        evaluator = ModelEvaluator()

        # Logistic Regression
        lr = self.train_logistic_regression(X_train, y_train)
        self.models["logistic_regression"] = lr
        self.results["logistic_regression"] = evaluator.evaluate(lr, X_test, y_test, "Logistic Regression")

        # Random Forest
        rf = self.train_random_forest(X_train, y_train)
        rf_calibrated = self.calibrate_model(rf, X_train, y_train)
        self.models["random_forest"] = rf_calibrated
        self.results["random_forest"] = evaluator.evaluate(rf_calibrated, X_test, y_test, "Random Forest")

        # XGBoost
        xgb = self.train_xgboost(X_train, y_train)
        if xgb is not None:
            xgb_calibrated = self.calibrate_model(xgb, X_train, y_train)
            self.models["xgboost"] = xgb_calibrated
            self.results["xgboost"] = evaluator.evaluate(xgb_calibrated, X_test, y_test, "XGBoost")

        # Select best model by AUC
        best_name = max(self.results, key=lambda k: self.results[k]["roc_auc"])
        self.best_model_name = best_name
        logger.info(f"Best model: {best_name} (AUC={self.results[best_name]['roc_auc']:.4f})")

        # Save best model
        best_model = self.models[best_name]
        joblib.dump(best_model, "models/best_model.pkl")
        joblib.dump(self.feature_names, "models/feature_names.pkl")
        logger.info("Saved best model to models/best_model.pkl")

        return {
            "models": self.models,
            "results": self.results,
            "best_model": best_name,
            "feature_names": self.feature_names,
            "test_data": {"X_test": X_test, "y_test": y_test},
        }
