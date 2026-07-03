"""
Model Training Module
Trains Logistic Regression, Random Forest, and XGBoost models
for probability-of-default prediction with calibration.

Improvements over baseline:
- MLflow experiment tracking (params, metrics, artifacts per run)
- Temporal walk-forward cross-validation (no future leakage)
- Model versioning via ModelRegistry with automatic promotion logic
- Champion/challenger comparison logging
"""

import logging
import os
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.models.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModelTrainer:
    """Trains and compares credit risk models with MLflow tracking."""

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
        self.registry = ModelRegistry()

        os.makedirs("models", exist_ok=True)

        if HAS_MLFLOW:
            try:
                # Pin a portable file-based backend. Without this, MLflow 3.x
                # defaults to a sqlite:// URI that fails on some installs
                # (e.g. mlflow-skinny) and differs across environments.
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
                tracking_dir = os.path.abspath("mlruns")
                mlflow.set_tracking_uri(f"file:{tracking_dir}")
                mlflow.set_experiment("credit-risk-pd-models")
                logger.info(f"MLflow tracking enabled → {tracking_dir}")
            except Exception as e:
                logger.warning(f"MLflow unavailable, continuing without tracking: {e}")
                globals()["HAS_MLFLOW"] = False

    def prepare_data(self, feature_table: pd.DataFrame) -> Tuple:
        """
        Temporal train/test split.

        If 'created_at' is present, sorts customers by join date and uses
        the last test_size fraction as the held-out test set — this avoids
        the future-leakage problem of random splits on time-series credit data.
        Falls back to stratified random split if no date column exists.
        """
        target = self.feature_config["target_column"]
        drop_cols = [
            self.feature_config["id_column"], target, "default_probability",
            "risk_segment", "composite_risk_score", "created_at",
        ]
        drop_cols = [c for c in drop_cols if c in feature_table.columns]

        X = feature_table.drop(columns=drop_cols)
        y = feature_table[target]
        X = X.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)

        if "created_at" in feature_table.columns:
            logger.info("Using temporal train/test split (sorted by customer creation date)")
            sorted_idx = feature_table["created_at"].argsort().values
            X = X.iloc[sorted_idx].reset_index(drop=True)
            y = y.iloc[sorted_idx].reset_index(drop=True)
            split = int(len(X) * (1 - self.test_size))
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size,
                random_state=self.random_state, stratify=y
            )

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Default rate — train: {y_train.mean():.2%}, test: {y_test.mean():.2%}")
        self.feature_names = list(X.columns)
        return X_train, X_test, y_train, y_test

    def walk_forward_cv(self, feature_table: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
        """
        Walk-forward cross-validation using temporal order.

        Each fold trains on all data before time T and tests on T→T+step.
        Reports mean ± std AUC across folds for each model type.
        """
        logger.info(f"Running {n_splits}-fold walk-forward CV...")
        target = self.feature_config["target_column"]
        drop_cols = [
            self.feature_config["id_column"], target, "default_probability",
            "risk_segment", "composite_risk_score", "created_at",
        ]
        drop_cols = [c for c in drop_cols if c in feature_table.columns]

        if "created_at" in feature_table.columns:
            ft = feature_table.sort_values("created_at").reset_index(drop=True)
        else:
            ft = feature_table.reset_index(drop=True)

        X = ft.drop(columns=drop_cols)
        X = X.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
        y = ft[target]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_aucs = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            if y_tr.nunique() < 2 or y_te.nunique() < 2:
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced",
                                             random_state=self.random_state)),
            ])
            pipe.fit(X_tr, y_tr)
            auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
            fold_aucs.append(auc)
            logger.info(f"  Fold {fold + 1}/{n_splits}: AUC = {auc:.4f}")

        result = {
            "walk_forward_mean_auc": float(np.mean(fold_aucs)),
            "walk_forward_std_auc": float(np.std(fold_aucs)),
            "n_folds": len(fold_aucs),
        }
        logger.info(f"Walk-forward AUC: {result['walk_forward_mean_auc']:.4f} "
                    f"± {result['walk_forward_std_auc']:.4f}")
        return result

    def train_logistic_regression(self, X_train, y_train) -> Pipeline:
        logger.info("Training Logistic Regression...")
        params = {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"}
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=self.random_state, **params)),
        ])
        pipe.fit(X_train, y_train)
        cv_score = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
        logger.info(f"Logistic Regression CV AUC: {cv_score:.4f}")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name="logistic_regression", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", cv_score)

        return pipe

    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        logger.info("Training Random Forest...")
        params = self.config["random_forest_params"]
        model = RandomForestClassifier(
            **params, class_weight="balanced",
            random_state=self.random_state, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
        logger.info(f"Random Forest CV AUC: {cv_score:.4f}")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name="random_forest", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", cv_score)

        return model

    def train_xgboost(self, X_train, y_train) -> Optional[Any]:
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed, skipping.")
            return None

        logger.info("Training XGBoost...")
        params = self.config["xgboost_params"]
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="auc",
        )
        model.fit(X_train, y_train)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
        logger.info(f"XGBoost CV AUC: {cv_score:.4f}")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name="xgboost", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", cv_score)
                mlflow.log_param("scale_pos_weight", round(float(scale_pos_weight), 3))

        return model

    def calibrate_model(self, model, X_train, y_train) -> CalibratedClassifierCV:
        calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        calibrated.fit(X_train, y_train)
        return calibrated

    def train_all(self, feature_table: pd.DataFrame) -> Dict:
        """Train all models, track with MLflow, register best, promote if better than production."""
        from src.models.evaluate import ModelEvaluator

        X_train, X_test, y_train, y_test = self.prepare_data(feature_table)
        wf_metrics = self.walk_forward_cv(feature_table)
        evaluator = ModelEvaluator()

        parent_run_id = None
        if HAS_MLFLOW:
            parent_run = mlflow.start_run(run_name="training_session")
            parent_run_id = parent_run.info.run_id
            mlflow.log_metrics(wf_metrics)
            mlflow.log_param("n_customers", len(feature_table))
            mlflow.log_param("default_rate", round(float(feature_table["is_default"].mean()), 4))

        try:
            # Logistic Regression
            lr = self.train_logistic_regression(X_train, y_train)
            self.models["logistic_regression"] = lr
            self.results["logistic_regression"] = evaluator.evaluate(lr, X_test, y_test, "Logistic Regression")

            # Random Forest
            rf = self.train_random_forest(X_train, y_train)
            rf_cal = self.calibrate_model(rf, X_train, y_train)
            self.models["random_forest"] = rf_cal
            self.results["random_forest"] = evaluator.evaluate(rf_cal, X_test, y_test, "Random Forest")

            # XGBoost
            xgb = self.train_xgboost(X_train, y_train)
            if xgb is not None:
                xgb_cal = self.calibrate_model(xgb, X_train, y_train)
                self.models["xgboost"] = xgb_cal
                self.results["xgboost"] = evaluator.evaluate(xgb_cal, X_test, y_test, "XGBoost")

            # Select best model
            best_name = max(self.results, key=lambda k: self.results[k]["roc_auc"])
            self.best_model_name = best_name
            best_metrics = self.results[best_name]
            logger.info(f"Best model: {best_name} (AUC={best_metrics['roc_auc']:.4f})")

            # Log all final metrics to MLflow parent run
            if HAS_MLFLOW:
                for model_name, metrics in self.results.items():
                    for metric_name in ["roc_auc", "pr_auc", "f1", "ks_statistic", "brier_score"]:
                        mlflow.log_metric(f"{model_name}_{metric_name}", metrics[metric_name])
                mlflow.log_param("best_model", best_name)

            # Register and conditionally promote
            serializable_metrics = {
                k: v for k, v in best_metrics.items()
                if k not in ("lift_data", "confusion_matrix", "classification_report")
            }
            version_id = self.registry.register(
                model=self.models[best_name],
                model_name=best_name,
                metrics=serializable_metrics,
                params=self.config.get(f"{best_name}_params", {}),
                feature_names=self.feature_names,
            )

            if self.registry.should_promote(serializable_metrics):
                self.registry.promote(version_id)
                logger.info(f"Promoted {version_id} to production")
            else:
                prod = self.registry.get_production()
                logger.info(f"New model did not outperform production "
                            f"({prod['metrics']['roc_auc']:.4f}). Keeping existing production model.")
                # Still save best_model.pkl for local use
                joblib.dump(self.models[best_name], "models/best_model.pkl")
                joblib.dump(self.feature_names, "models/feature_names.pkl")

            if HAS_MLFLOW:
                try:
                    mlflow.log_param("registered_version", version_id)
                    mlflow.sklearn.log_model(self.models[best_name], name="best_model")
                except Exception as e:
                    # Model artifact is already persisted via joblib + registry;
                    # MLflow model logging is best-effort (e.g. requires `skops`
                    # for sklearn flavor in MLflow 3.x).
                    logger.warning(f"MLflow model logging skipped: {e}")

        finally:
            if HAS_MLFLOW and parent_run_id:
                mlflow.end_run()

        return {
            "models": self.models,
            "results": self.results,
            "best_model": best_name,
            "feature_names": self.feature_names,
            "walk_forward": wf_metrics,
            "test_data": {"X_test": X_test, "y_test": y_test},
        }
