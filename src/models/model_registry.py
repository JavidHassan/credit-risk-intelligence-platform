"""
Model Registry
Handles model versioning, metadata persistence, and production promotion.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import joblib

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Versioned model registry with promotion logic."""

    def __init__(self, registry_dir: str = "models"):
        self.registry_dir = registry_dir
        self.registry_path = os.path.join(registry_dir, "registry.json")
        os.makedirs(registry_dir, exist_ok=True)
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                self.registry = json.load(f)
        else:
            self.registry = {"versions": [], "production": None}

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register(
        self,
        model,
        model_name: str,
        metrics: Dict,
        params: Dict,
        feature_names: List[str],
        data_hash: Optional[str] = None,
    ) -> str:
        """Register a new model version."""
        version = len(self.registry["versions"]) + 1
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        version_id = f"v{version}_{timestamp}_{model_name}"
        version_dir = os.path.join(self.registry_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)

        # Save model artifact
        model_path = os.path.join(version_dir, "model.pkl")
        joblib.dump(model, model_path)
        joblib.dump(feature_names, os.path.join(version_dir, "feature_names.pkl"))

        # Save metadata
        metadata = {
            "version_id": version_id,
            "version": version,
            "model_name": model_name,
            "timestamp": timestamp,
            "metrics": metrics,
            "params": params,
            "feature_names": feature_names,
            "data_hash": data_hash,
            "status": "staging",
        }
        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.registry["versions"].append(metadata)
        self._save_registry()
        logger.info(f"Registered model: {version_id} (AUC={metrics.get('roc_auc', 'N/A'):.4f})")
        return version_id

    def promote(self, version_id: str):
        """Promote a model version to production."""
        # Update status in registry
        for v in self.registry["versions"]:
            if v["version_id"] == version_id:
                v["status"] = "production"
                v["promoted_at"] = datetime.utcnow().isoformat()
            elif v["status"] == "production":
                v["status"] = "archived"

        self.registry["production"] = version_id
        self._save_registry()

        # Copy model to canonical path for API
        version_dir = os.path.join(self.registry_dir, version_id)
        shutil.copy(
            os.path.join(version_dir, "model.pkl"),
            os.path.join(self.registry_dir, "best_model.pkl"),
        )
        shutil.copy(
            os.path.join(version_dir, "feature_names.pkl"),
            os.path.join(self.registry_dir, "feature_names.pkl"),
        )
        logger.info(f"Promoted {version_id} to production")

    def get_production(self) -> Optional[Dict]:
        """Return metadata for the current production model."""
        prod_id = self.registry.get("production")
        if not prod_id:
            return None
        for v in self.registry["versions"]:
            if v["version_id"] == prod_id:
                return v
        return None

    def list_versions(self) -> List[Dict]:
        return self.registry["versions"]

    def should_promote(self, new_metrics: Dict, min_improvement: float = 0.005) -> bool:
        """Return True if new model is meaningfully better than production."""
        prod = self.get_production()
        if prod is None:
            return True
        prod_auc = prod["metrics"].get("roc_auc", 0)
        new_auc = new_metrics.get("roc_auc", 0)
        improvement = new_auc - prod_auc
        logger.info(f"AUC comparison — prod: {prod_auc:.4f}, new: {new_auc:.4f}, delta: {improvement:+.4f}")
        return improvement >= min_improvement
