"""
Data Preprocessing Pipeline
Raw data cleaning, validation, missing-value handling, and outlier treatment.
"""

import logging
import pandas as pd
from typing import Dict, List
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates raw data quality before processing."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["preprocessing"]
        self.missing_threshold = self.config["missing_threshold"]
        self.validation_report: List[Dict] = []

    def check_missing_values(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Flag and report missing values."""
        missing = df.isnull().sum()
        missing_pct = missing / len(df)
        high_missing = missing_pct[missing_pct > self.missing_threshold]

        if len(high_missing) > 0:
            logger.warning(
                f"[{name}] Columns with >{self.missing_threshold:.0%} missing: "
                f"{dict(high_missing.round(3))}"
            )

        self.validation_report.append({
            "dataset": name,
            "total_rows": len(df),
            "columns_with_missing": int((missing > 0).sum()),
            "high_missing_columns": list(high_missing.index),
        })
        return df

    def check_duplicates(self, df: pd.DataFrame, key_col: str, name: str) -> pd.DataFrame:
        """Remove duplicate rows by key column."""
        n_dupes = df.duplicated(subset=[key_col]).sum()
        if n_dupes > 0:
            logger.warning(f"[{name}] Removing {n_dupes} duplicate rows on '{key_col}'")
            df = df.drop_duplicates(subset=[key_col], keep="first")
        return df

    def check_value_ranges(self, df: pd.DataFrame, range_rules: Dict, name: str) -> pd.DataFrame:
        """Validate numeric columns fall within expected ranges."""
        for col, (lo, hi) in range_rules.items():
            if col in df.columns:
                out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
                if out_of_range > 0:
                    logger.warning(
                        f"[{name}] {out_of_range} values in '{col}' outside [{lo}, {hi}] — clipping"
                    )
                    df[col] = df[col].clip(lo, hi)
        return df


class DataPreprocessor:
    """Cleans and transforms raw data into analysis-ready tables."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.config = cfg["preprocessing"]
        self.outlier_std = self.config["outlier_std"]
        self.validator = DataValidator(config_path)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with appropriate strategies."""
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            if df[col].dtype in ["float64", "int64"]:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed '{col}' with median={median_val:.2f}")
            elif df[col].dtype == "object":
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown"
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Imputed '{col}' with mode='{mode_val}'")
        return df

    def treat_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Winsorize outliers beyond N standard deviations."""
        for col in numeric_cols:
            if col not in df.columns:
                continue
            mean, std = df[col].mean(), df[col].std()
            lower, upper = mean - self.outlier_std * std, mean + self.outlier_std * std
            n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_clipped > 0:
                df[col] = df[col].clip(lower, upper)
                logger.info(f"Winsorized {n_clipped} outliers in '{col}'")
        return df

    def encode_categoricals(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        for col in columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    def preprocess_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing for customer data."""
        df = self.validator.check_missing_values(df, "customers")
        df = self.validator.check_duplicates(df, "customer_id", "customers")
        df = self.validator.check_value_ranges(df, {
            "age": (18, 100), "annual_income": (0, 1_000_000),
            "credit_score": (300, 850), "dependents": (0, 10),
        }, "customers")

        df = self.handle_missing_values(df)
        df = self.treat_outliers(df, ["annual_income", "employment_years"])
        df = self.encode_categoricals(df, [
            "gender", "employment_status", "education", "marital_status", "state"
        ])
        return df

    def preprocess_statements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing for statement data."""
        df = self.validator.check_missing_values(df, "statements")
        df["statement_date"] = pd.to_datetime(df["statement_date"])
        df = self.handle_missing_values(df)
        df = self.treat_outliers(df, ["statement_balance", "total_spend", "total_payment"])
        return df

    def preprocess_all(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Run preprocessing on all raw datasets."""
        logger.info("Starting preprocessing pipeline...")
        processed = {}
        processed["customers"] = self.preprocess_customers(datasets["customers"])
        processed["statements"] = self.preprocess_statements(datasets["statements"])

        for name in ["accounts", "credit_cards", "transactions", "payments",
                     "delinquencies", "defaults", "macro_variables"]:
            if name in datasets:
                df = datasets[name].copy()
                df = self.validator.check_missing_values(df, name)
                df = self.handle_missing_values(df)
                processed[name] = df

        logger.info("Preprocessing complete.")
        return processed
