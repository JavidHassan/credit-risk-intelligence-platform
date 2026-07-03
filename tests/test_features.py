"""Tests for feature engineering pipeline."""

import pytest
import numpy as np
import pandas as pd
from src.pipelines.feature_pipeline import CreditFeatureEngineer


@pytest.fixture
def engineer():
    return CreditFeatureEngineer.__new__(CreditFeatureEngineer)


@pytest.fixture(autouse=True)
def set_config(engineer):
    engineer.rolling_windows = [3, 6, 12]
    engineer.target_col = "is_default"
    engineer.id_col = "customer_id"


@pytest.fixture
def sample_statements():
    np.random.seed(42)
    records = []
    for cust in ["CUST_00001", "CUST_00002"]:
        for month in range(12):
            records.append({
                "customer_id": cust,
                "card_id": f"CARD_{cust[-5:]}",
                "statement_date": pd.Timestamp("2021-01-01") + pd.DateOffset(months=month),
                "statement_balance": np.random.uniform(500, 5000),
                "total_spend": np.random.uniform(200, 2000),
                "total_payment": np.random.uniform(100, 1500),
                "minimum_payment": 25.0,
                "credit_limit": 10000,
                "utilization": np.random.uniform(0.1, 0.8),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_payments():
    return pd.DataFrame({
        "customer_id": ["CUST_00001"] * 10 + ["CUST_00002"] * 10,
        "card_id": ["CARD_00001"] * 10 + ["CARD_00002"] * 10,
        "payment_date": pd.date_range("2021-01-15", periods=20, freq="MS"),
        "payment_amount": np.random.uniform(100, 1000, 20),
        "payment_method": ["auto_pay"] * 20,
        "is_late": [False] * 8 + [True] * 2 + [False] * 7 + [True] * 3,
    })


class TestUtilizationFeatures:
    def test_output_columns(self, engineer, sample_statements):
        result = engineer.compute_utilization_features(sample_statements)
        expected_cols = ["customer_id", "avg_utilization", "max_utilization",
                         "min_utilization", "std_utilization", "current_utilization"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_utilization_range(self, engineer, sample_statements):
        result = engineer.compute_utilization_features(sample_statements)
        assert result["avg_utilization"].between(0, 1).all()
        assert result["max_utilization"].between(0, 1).all()

    def test_customer_count(self, engineer, sample_statements):
        result = engineer.compute_utilization_features(sample_statements)
        assert len(result) == 2


class TestPaymentFeatures:
    def test_late_payment_count(self, engineer, sample_statements, sample_payments):
        result = engineer.compute_payment_features(sample_statements, sample_payments)
        assert "late_payment_count" in result.columns
        assert result["late_payment_count"].min() >= 0

    def test_payment_ratio(self, engineer, sample_statements, sample_payments):
        result = engineer.compute_payment_features(sample_statements, sample_payments)
        assert "late_payment_ratio" in result.columns
        assert result["late_payment_ratio"].between(0, 1).all()


class TestDelinquencyFeatures:
    def test_empty_delinquencies(self, engineer):
        empty = pd.DataFrame(columns=[
            "customer_id", "late_count", "delinquency_status", "delinquency_amount"
        ])
        result = engineer.compute_delinquency_features(empty)
        assert len(result) == 0

    def test_severity_mapping(self, engineer):
        delinq = pd.DataFrame({
            "customer_id": ["CUST_00001", "CUST_00002"],
            "late_count": [3, 7],
            "delinquency_status": ["30_dpd", "90_dpd"],
            "delinquency_amount": [1000, 5000],
        })
        result = engineer.compute_delinquency_features(delinq)
        assert result["delinquency_severity"].tolist() == [1, 3]
