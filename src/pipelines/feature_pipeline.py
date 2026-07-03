"""
Feature Engineering Pipeline
Computes behavioral credit features from preprocessed data.
Supports PySpark for scalable processing and Pandas for local development.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditFeatureEngineer:
    """Engineers credit risk features from banking data."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.rolling_windows = cfg["features"]["rolling_windows"]
        self.target_col = cfg["features"]["target_column"]
        self.id_col = cfg["features"]["id_column"]

    def compute_utilization_features(self, statements: pd.DataFrame) -> pd.DataFrame:
        """Credit utilization metrics."""
        logger.info("Computing utilization features...")
        grouped = statements.groupby("customer_id")

        features = pd.DataFrame({
            "customer_id": grouped["customer_id"].first(),
            "avg_utilization": grouped["utilization"].mean(),
            "max_utilization": grouped["utilization"].max(),
            "min_utilization": grouped["utilization"].min(),
            "std_utilization": grouped["utilization"].std().fillna(0),
            "current_utilization": grouped["utilization"].last(),
            "utilization_trend": grouped["utilization"].apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            ),
        })
        return features.reset_index(drop=True)

    def compute_payment_features(self, statements: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
        """Payment behavior metrics."""
        logger.info("Computing payment features...")
        stmt_grouped = statements.groupby("customer_id")

        payment_ratio = statements.copy()
        payment_ratio["payment_to_balance"] = np.where(
            payment_ratio["statement_balance"] > 0,
            payment_ratio["total_payment"] / payment_ratio["statement_balance"],
            1.0
        )

        pr_grouped = payment_ratio.groupby("customer_id")
        late_counts = payments.groupby("customer_id")["is_late"].sum().reset_index()
        late_counts.columns = ["customer_id", "late_payment_count"]

        total_payments = payments.groupby("customer_id").size().reset_index(name="total_payment_count")

        features = pd.DataFrame({
            "customer_id": pr_grouped["customer_id"].first(),
            "avg_payment_to_balance": pr_grouped["payment_to_balance"].mean(),
            "min_payment_to_balance": pr_grouped["payment_to_balance"].min(),
            "payment_volatility": pr_grouped["total_payment"].std().fillna(0),
            "avg_monthly_payment": pr_grouped["total_payment"].mean(),
        }).reset_index(drop=True)

        features = features.merge(late_counts, on="customer_id", how="left")
        features = features.merge(total_payments, on="customer_id", how="left")
        features["late_payment_count"] = features["late_payment_count"].fillna(0)
        features["late_payment_ratio"] = np.where(
            features["total_payment_count"] > 0,
            features["late_payment_count"] / features["total_payment_count"],
            0
        )
        return features

    def compute_spending_features(self, statements: pd.DataFrame) -> pd.DataFrame:
        """Rolling spending and trend metrics."""
        logger.info("Computing spending features...")
        statements = statements.sort_values(["customer_id", "statement_date"])

        result_frames = []
        for window in self.rolling_windows:
            rolled = statements.groupby("customer_id")["total_spend"].rolling(
                window, min_periods=1
            ).agg(["mean", "std"])
            rolled.columns = [f"spend_mean_{window}m", f"spend_std_{window}m"]
            rolled = rolled.reset_index(level=0)
            result_frames.append(rolled.groupby("customer_id").last())

        features = result_frames[0]
        for rf in result_frames[1:]:
            features = features.join(rf, how="outer")

        features["spend_trend"] = statements.groupby("customer_id")["total_spend"].apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        return features.fillna(0).reset_index()

    def compute_delinquency_features(self, delinquencies: pd.DataFrame) -> pd.DataFrame:
        """Delinquency history features."""
        logger.info("Computing delinquency features...")
        severity_map = {"30_dpd": 1, "60_dpd": 2, "90_dpd": 3, "120_plus_dpd": 4}

        if len(delinquencies) == 0:
            return pd.DataFrame(columns=[
                "customer_id", "delinquency_severity", "delinquency_amount",
                "is_delinquent", "late_count"
            ])

        features = delinquencies.copy()
        features["delinquency_severity"] = features["delinquency_status"].map(severity_map).fillna(0)
        features["is_delinquent"] = 1
        return features[["customer_id", "delinquency_severity", "delinquency_amount",
                         "is_delinquent", "late_count"]]

    def compute_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Transaction-level features: category risk, merchant patterns."""
        logger.info("Computing transaction features...")
        merchant_risk_map = {"low": 0, "medium": 1, "high": 2}
        transactions["merchant_risk_score"] = transactions["merchant_risk"].map(merchant_risk_map)

        grouped = transactions.groupby("customer_id")
        features = pd.DataFrame({
            "customer_id": grouped["customer_id"].first(),
            "avg_transaction_amount": grouped["amount"].mean(),
            "max_transaction_amount": grouped["amount"].max(),
            "total_transaction_count": grouped.size(),
            "avg_merchant_risk": grouped["merchant_risk_score"].mean(),
            "high_risk_txn_ratio": grouped["merchant_risk_score"].apply(
                lambda x: (x == 2).mean()
            ),
            "international_txn_ratio": grouped["is_international"].mean(),
            "n_categories": grouped["category"].nunique(),
        }).reset_index(drop=True)
        return features

    def compute_income_debt_features(
        self, customers: pd.DataFrame, statements: pd.DataFrame
    ) -> pd.DataFrame:
        """Income-to-debt ratio and related features."""
        logger.info("Computing income-debt features...")
        latest_balance = statements.sort_values("statement_date").groupby(
            "customer_id"
        )["statement_balance"].last().reset_index()
        latest_balance.columns = ["customer_id", "latest_balance"]

        income_cols = [c for c in customers.columns if c in ["customer_id", "annual_income"]]
        features = customers[income_cols].merge(latest_balance, on="customer_id", how="left")
        features["latest_balance"] = features["latest_balance"].fillna(0)
        features["income_to_debt_ratio"] = np.where(
            features["latest_balance"] > 0,
            features["annual_income"] / (features["latest_balance"] * 12),
            10.0
        )
        features["monthly_income"] = features["annual_income"] / 12
        features["debt_burden"] = features["latest_balance"] / features["monthly_income"]
        return features[["customer_id", "income_to_debt_ratio", "debt_burden", "latest_balance"]]

    def compute_risk_segmentation(self, feature_table: pd.DataFrame) -> pd.DataFrame:
        """Assign customer risk segments based on composite score."""
        logger.info("Computing risk segmentation...")
        df = feature_table.copy()

        risk_score = (
            df.get("avg_utilization", 0) * 0.25
            + df.get("late_payment_ratio", 0) * 0.25
            + df.get("delinquency_severity", 0).fillna(0) / 4 * 0.20
            + (1 - df.get("avg_payment_to_balance", 1).clip(0, 1)) * 0.15
            + df.get("avg_merchant_risk", 0) / 2 * 0.15
        )

        df["composite_risk_score"] = risk_score.round(4)
        df["risk_segment"] = pd.cut(
            risk_score, bins=[-np.inf, 0.2, 0.4, 0.6, np.inf],
            labels=["low", "medium", "high", "critical"]
        )
        return df

    def build_feature_table(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build the complete feature table from all data sources."""
        logger.info("Building feature table...")

        utilization = self.compute_utilization_features(datasets["statements"])
        payments = self.compute_payment_features(datasets["statements"], datasets["payments"])
        spending = self.compute_spending_features(datasets["statements"])
        delinquency = self.compute_delinquency_features(datasets["delinquencies"])
        transactions = self.compute_transaction_features(datasets["transactions"])
        income_debt = self.compute_income_debt_features(
            datasets["customers"], datasets["statements"]
        )

        feature_table = utilization
        for df in [payments, spending, delinquency, transactions, income_debt]:
            feature_table = feature_table.merge(df, on="customer_id", how="left")

        # Add customer demographic and bureau features (strong predictors)
        customer_cols = ["customer_id", "credit_score", "age", "annual_income",
                         "employment_years", "dependents"]
        available_cols = [c for c in customer_cols if c in datasets["customers"].columns]
        feature_table = feature_table.merge(
            datasets["customers"][available_cols], on="customer_id", how="left"
        )

        feature_table = feature_table.merge(
            datasets["defaults"][["customer_id", "is_default", "default_probability"]],
            on="customer_id", how="left"
        )

        # Fill missing delinquency features for non-delinquent customers
        feature_table["is_delinquent"] = feature_table["is_delinquent"].fillna(0)
        feature_table["delinquency_severity"] = feature_table["delinquency_severity"].fillna(0)
        feature_table["delinquency_amount"] = feature_table["delinquency_amount"].fillna(0)
        feature_table["late_count"] = feature_table["late_count"].fillna(0)

        feature_table = self.compute_risk_segmentation(feature_table)

        logger.info(f"Feature table shape: {feature_table.shape}")
        logger.info(f"Default rate: {feature_table['is_default'].mean():.2%}")
        return feature_table
