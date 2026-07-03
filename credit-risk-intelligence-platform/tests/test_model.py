"""Tests for model training and evaluation."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from src.models.evaluate import ModelEvaluator


@pytest.fixture
def evaluator():
    return ModelEvaluator()


@pytest.fixture
def trained_model():
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        random_state=42, weights=[0.9, 0.1]
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model, pd.DataFrame(X), pd.Series(y)


class TestModelEvaluator:
    def test_evaluate_returns_metrics(self, evaluator, trained_model):
        model, X, y = trained_model
        result = evaluator.evaluate(model, X, y, "Test Model")
        assert "roc_auc" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "ks_statistic" in result
        assert "brier_score" in result

    def test_auc_range(self, evaluator, trained_model):
        model, X, y = trained_model
        result = evaluator.evaluate(model, X, y, "Test Model")
        assert 0 <= result["roc_auc"] <= 1

    def test_ks_statistic_range(self, evaluator, trained_model):
        model, X, y = trained_model
        result = evaluator.evaluate(model, X, y, "Test Model")
        assert 0 <= result["ks_statistic"] <= 1

    def test_confusion_matrix_shape(self, evaluator, trained_model):
        model, X, y = trained_model
        result = evaluator.evaluate(model, X, y, "Test Model")
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_lift_data(self, evaluator, trained_model):
        model, X, y = trained_model
        result = evaluator.evaluate(model, X, y, "Test Model")
        assert "lift_data" in result
        assert len(result["lift_data"]) > 0

    def test_compare_models(self, evaluator, trained_model):
        model, X, y = trained_model
        r1 = evaluator.evaluate(model, X, y, "Model A")
        r2 = evaluator.evaluate(model, X, y, "Model B")
        comparison = evaluator.compare_models({"model_a": r1, "model_b": r2})
        assert len(comparison) == 2
        assert "roc_auc" in comparison.columns
