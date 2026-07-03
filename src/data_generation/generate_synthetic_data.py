"""
Synthetic Bank Data Generator
Generates realistic customer, account, credit card, transaction,
statement, payment, delinquency, default, and macroeconomic data.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticBankDataGenerator:
    """Generates synthetic banking data for credit risk modeling."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["data_generation"]

        self.n_customers = self.config["n_customers"]
        self.n_months = self.config["n_months"]
        self.default_rate = self.config["default_rate"]
        self.seed = self.config["seed"]
        np.random.seed(self.seed)
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer demographics."""
        logger.info("Generating customer data...")
        ages = np.random.normal(42, 14, self.n_customers).clip(18, 80).astype(int)
        income = np.random.lognormal(10.8, 0.7, self.n_customers).clip(15000, 500000).round(2)
        employment_years = (ages - 18) * np.random.uniform(0.1, 0.8, self.n_customers)
        employment_years = employment_years.clip(0, 40).round(1)

        customers = pd.DataFrame({
            "customer_id": [f"CUST_{i:05d}" for i in range(self.n_customers)],
            "age": ages,
            "gender": np.random.choice(["M", "F", "Other"], self.n_customers, p=[0.48, 0.48, 0.04]),
            "annual_income": income,
            "employment_status": np.random.choice(
                ["employed", "self_employed", "unemployed", "retired"],
                self.n_customers, p=[0.60, 0.20, 0.10, 0.10]
            ),
            "employment_years": employment_years,
            "education": np.random.choice(
                ["high_school", "bachelors", "masters", "phd", "other"],
                self.n_customers, p=[0.30, 0.35, 0.20, 0.05, 0.10]
            ),
            "marital_status": np.random.choice(
                ["single", "married", "divorced", "widowed"],
                self.n_customers, p=[0.30, 0.45, 0.20, 0.05]
            ),
            "dependents": np.random.poisson(1.2, self.n_customers).clip(0, 6),
            "state": np.random.choice(
                ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"],
                self.n_customers
            ),
            "credit_score": np.random.normal(680, 80, self.n_customers).clip(300, 850).astype(int),
            "created_at": pd.to_datetime("2020-01-01") + pd.to_timedelta(
                np.random.randint(0, 365, self.n_customers), unit="D"
            ),
        })
        return customers

    def generate_accounts(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Generate bank accounts linked to customers."""
        logger.info("Generating account data...")
        accounts = pd.DataFrame({
            "account_id": [f"ACCT_{i:05d}" for i in range(self.n_customers)],
            "customer_id": customers["customer_id"],
            "account_type": np.random.choice(
                ["checking", "savings", "money_market"],
                self.n_customers, p=[0.50, 0.35, 0.15]
            ),
            "balance": np.random.lognormal(8.5, 1.5, self.n_customers).clip(100, 200000).round(2),
            "opened_date": customers["created_at"],
            "status": np.random.choice(["active", "inactive", "closed"],
                                       self.n_customers, p=[0.85, 0.10, 0.05]),
        })
        return accounts

    def generate_credit_cards(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Generate credit card data with limits based on income and credit score."""
        logger.info("Generating credit card data...")
        income = customers["annual_income"].values
        credit_score = customers["credit_score"].values

        credit_limit = (income * 0.3 + (credit_score - 300) * 50).clip(500, 100000).round(-2)
        apr = (30 - credit_score / 40 + np.random.normal(0, 2, self.n_customers)).clip(8, 30).round(2)

        cards = pd.DataFrame({
            "card_id": [f"CARD_{i:05d}" for i in range(self.n_customers)],
            "customer_id": customers["customer_id"],
            "credit_limit": credit_limit,
            "apr": apr,
            "card_type": np.random.choice(
                ["standard", "gold", "platinum", "rewards"],
                self.n_customers, p=[0.40, 0.30, 0.15, 0.15]
            ),
            "issued_date": customers["created_at"] + pd.to_timedelta(
                np.random.randint(0, 90, self.n_customers), unit="D"
            ),
        })
        return cards

    def generate_monthly_statements(
        self, customers: pd.DataFrame, cards: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate monthly credit card statements."""
        logger.info("Generating monthly statements...")
        records = []
        base_date = pd.to_datetime("2020-04-01")

        for i in range(self.n_customers):
            cust_id = customers.iloc[i]["customer_id"]
            card_id = cards.iloc[i]["card_id"]
            credit_limit = cards.iloc[i]["credit_limit"]
            income = customers.iloc[i]["annual_income"]

            monthly_spend_base = income / 12 * np.random.uniform(0.15, 0.45)
            balance = credit_limit * np.random.uniform(0.1, 0.5)

            for month in range(self.n_months):
                statement_date = base_date + pd.DateOffset(months=month)
                spend = monthly_spend_base * np.random.lognormal(0, 0.3)
                payment = spend * np.random.uniform(0.3, 1.2)
                balance = max(0, balance + spend - payment)
                min_payment = max(25, balance * 0.02)

                records.append({
                    "customer_id": cust_id,
                    "card_id": card_id,
                    "statement_date": statement_date,
                    "statement_balance": round(balance, 2),
                    "total_spend": round(spend, 2),
                    "total_payment": round(payment, 2),
                    "minimum_payment": round(min_payment, 2),
                    "credit_limit": credit_limit,
                    "utilization": round(balance / credit_limit, 4),
                })

        return pd.DataFrame(records)

    def generate_transactions(self, customers: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
        """Generate individual credit card transactions."""
        logger.info("Generating transactions...")
        categories = [
            ("groceries", 80, 30), ("gas", 55, 15), ("restaurants", 45, 20),
            ("online_shopping", 120, 60), ("utilities", 150, 40),
            ("entertainment", 60, 30), ("travel", 500, 300),
            ("healthcare", 200, 100), ("education", 300, 150),
        ]

        records = []
        base_date = pd.to_datetime("2020-04-01")
        end_date = base_date + pd.DateOffset(months=self.n_months)

        for i in range(min(self.n_customers, 2000)):
            cust_id = customers.iloc[i]["customer_id"]
            card_id = cards.iloc[i]["card_id"]
            n_txns = np.random.randint(5, 30) * self.n_months

            for _ in range(n_txns):
                cat, mean_amt, std_amt = categories[np.random.randint(len(categories))]
                amount = max(1, np.random.normal(mean_amt, std_amt))
                txn_date = base_date + timedelta(days=np.random.randint(0, self.n_months * 30))

                if txn_date < end_date:
                    records.append({
                        "transaction_id": f"TXN_{len(records):08d}",
                        "customer_id": cust_id,
                        "card_id": card_id,
                        "transaction_date": txn_date,
                        "amount": round(amount, 2),
                        "category": cat,
                        "merchant_risk": np.random.choice(
                            ["low", "medium", "high"], p=[0.6, 0.3, 0.1]
                        ),
                        "is_international": np.random.random() < 0.05,
                    })

        return pd.DataFrame(records)

    def generate_payments(self, statements: pd.DataFrame) -> pd.DataFrame:
        """Generate payment records tied to monthly statements."""
        logger.info("Generating payment data...")
        records = []
        for _, row in statements.iterrows():
            if np.random.random() < 0.92:
                days_after = np.random.choice(
                    [5, 10, 15, 20, 25, 30, 45],
                    p=[0.10, 0.25, 0.30, 0.15, 0.10, 0.05, 0.05]
                )
                pay_frac = np.random.choice(
                    [0.02, 0.10, 0.25, 0.50, 1.0],
                    p=[0.10, 0.15, 0.20, 0.25, 0.30]
                )
                amount = round(row["statement_balance"] * pay_frac, 2)

                records.append({
                    "customer_id": row["customer_id"],
                    "card_id": row["card_id"],
                    "payment_date": row["statement_date"] + timedelta(days=int(days_after)),
                    "payment_amount": max(amount, 0),
                    "payment_method": np.random.choice(
                        ["auto_pay", "online", "check", "phone"],
                        p=[0.40, 0.35, 0.15, 0.10]
                    ),
                    "is_late": days_after > 30,
                })

        return pd.DataFrame(records)

    def generate_delinquencies(self, customers: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
        """Generate delinquency records based on late payments."""
        logger.info("Generating delinquency data...")
        late_payments = payments[payments["is_late"]].copy()
        customer_late_counts = late_payments.groupby("customer_id").size().reset_index(name="late_count")

        delinquent = customer_late_counts[customer_late_counts["late_count"] >= 2].copy()
        delinquent["delinquency_status"] = delinquent["late_count"].apply(
            lambda x: "30_dpd" if x < 4 else ("60_dpd" if x < 6 else ("90_dpd" if x < 8 else "120_plus_dpd"))
        )
        delinquent["delinquency_amount"] = np.random.lognormal(7, 1, len(delinquent)).clip(200, 50000).round(2)

        return delinquent[["customer_id", "late_count", "delinquency_status", "delinquency_amount"]]

    def generate_defaults(self, customers: pd.DataFrame, delinquencies: pd.DataFrame) -> pd.DataFrame:
        """Generate default labels combining risk factors."""
        logger.info("Generating default labels...")
        df = customers[["customer_id", "credit_score", "annual_income", "employment_status"]].copy()

        df = df.merge(
            delinquencies[["customer_id", "late_count", "delinquency_status"]],
            on="customer_id", how="left"
        )
        df["late_count"] = df["late_count"].fillna(0)

        base_prob = 1 / (1 + np.exp(-(
            -4.0
            - 0.020 * (df["credit_score"] - 680)
            + 0.65 * df["late_count"]
            - 0.00003 * df["annual_income"]
            + 1.2 * (df["employment_status"] == "unemployed").astype(int)
        )))

        scale = self.default_rate / base_prob.mean()
        adjusted_prob = (base_prob * scale).clip(0, 0.95)
        df["default_probability"] = adjusted_prob.round(4)
        df["is_default"] = (np.random.random(len(df)) < adjusted_prob).astype(int)

        logger.info(f"Default rate: {df['is_default'].mean():.2%}")
        return df[["customer_id", "default_probability", "is_default"]]

    def generate_macro_variables(self) -> pd.DataFrame:
        """Generate macroeconomic time series."""
        logger.info("Generating macro variables...")
        base_date = pd.to_datetime("2020-04-01")
        dates = [base_date + pd.DateOffset(months=m) for m in range(self.n_months)]

        unemployment = np.cumsum(np.random.normal(0, 0.002, self.n_months)) + 0.05
        gdp_growth = np.cumsum(np.random.normal(0.002, 0.005, self.n_months))
        inflation = np.cumsum(np.random.normal(0, 0.001, self.n_months)) + 0.03
        fed_rate = np.cumsum(np.random.normal(0, 0.001, self.n_months)) + 0.02
        consumer_confidence = np.cumsum(np.random.normal(0, 2, self.n_months)) + 100

        return pd.DataFrame({
            "date": dates,
            "unemployment_rate": unemployment.clip(0.02, 0.15).round(4),
            "gdp_growth": gdp_growth.round(4),
            "inflation_rate": inflation.clip(0, 0.12).round(4),
            "fed_funds_rate": fed_rate.clip(0, 0.08).round(4),
            "consumer_confidence_index": consumer_confidence.clip(50, 150).round(1),
        })

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets and save to CSV."""
        logger.info("Starting synthetic data generation...")

        customers = self.generate_customers()
        accounts = self.generate_accounts(customers)
        cards = self.generate_credit_cards(customers)
        statements = self.generate_monthly_statements(customers, cards)
        transactions = self.generate_transactions(customers, cards)
        payments = self.generate_payments(statements)
        delinquencies = self.generate_delinquencies(customers, payments)
        defaults = self.generate_defaults(customers, delinquencies)
        macro = self.generate_macro_variables()

        datasets = {
            "customers": customers,
            "accounts": accounts,
            "credit_cards": cards,
            "statements": statements,
            "transactions": transactions,
            "payments": payments,
            "delinquencies": delinquencies,
            "defaults": defaults,
            "macro_variables": macro,
        }

        for name, df in datasets.items():
            path = os.path.join(self.output_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            logger.info(f"Saved {name}: {df.shape[0]} rows → {path}")

        logger.info("Data generation complete.")
        return datasets


if __name__ == "__main__":
    generator = SyntheticBankDataGenerator()
    datasets = generator.generate_all()
    for name, df in datasets.items():
        print(f"{name}: {df.shape}")
