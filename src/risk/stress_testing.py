"""
Stress Testing Module
Simulates portfolio impact under adverse macroeconomic scenarios.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressTester:
    """Runs stress test scenarios on the credit portfolio."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.scenarios = cfg["risk"]["stress_scenarios"]
        self.lgd_mean = cfg["risk"]["lgd_mean"]
        self.ead_factor = cfg["risk"]["ead_utilization_factor"]

    def apply_stress(
        self, feature_table: pd.DataFrame, scenario_name: str
    ) -> pd.DataFrame:
        """Apply a stress scenario to the feature table."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. "
                             f"Available: {list(self.scenarios.keys())}")

        scenario = self.scenarios[scenario_name]
        stressed = feature_table.copy()

        # Increase delinquency-related features
        delinq_increase = scenario["delinquency_increase"]
        if "late_payment_ratio" in stressed.columns:
            stressed["late_payment_ratio"] = (
                stressed["late_payment_ratio"] + delinq_increase
            ).clip(0, 1)

        if "delinquency_severity" in stressed.columns:
            stressed["delinquency_severity"] = (
                stressed["delinquency_severity"] + delinq_increase * 10
            ).clip(0, 4)

        # Decrease income-related features
        income_decrease = scenario["income_decrease"]
        if "income_to_debt_ratio" in stressed.columns:
            stressed["income_to_debt_ratio"] *= (1 - income_decrease)

        if "debt_burden" in stressed.columns:
            stressed["debt_burden"] *= (1 + income_decrease)

        # Increase utilization under stress
        unemployment_increase = scenario["unemployment_increase"]
        if "avg_utilization" in stressed.columns:
            stressed["avg_utilization"] = (
                stressed["avg_utilization"] + unemployment_increase * 2
            ).clip(0, 1)

        if "current_utilization" in stressed.columns:
            stressed["current_utilization"] = (
                stressed["current_utilization"] + unemployment_increase * 2
            ).clip(0, 1)

        # Stress the dominant bureau/behavioral drivers directly:
        # recessions push credit scores down, incomes down, and late payments up
        if "credit_score" in stressed.columns:
            score_drop = unemployment_increase * 400  # e.g. 10% unemployment shock → -40 pts
            stressed["credit_score"] = (stressed["credit_score"] - score_drop).clip(300, 850)

        if "annual_income" in stressed.columns:
            stressed["annual_income"] *= (1 - income_decrease)

        if "late_count" in stressed.columns:
            stressed["late_count"] = stressed["late_count"] + delinq_increase * 20

        if "late_payment_count" in stressed.columns:
            stressed["late_payment_count"] = (
                stressed["late_payment_count"] + delinq_increase * 20
            )

        logger.info(f"Applied '{scenario_name}' stress scenario")
        return stressed

    def run_stress_test(
        self, model, feature_table: pd.DataFrame, feature_names: list,
        credit_limits: pd.Series, current_balances: pd.Series,
    ) -> Dict:
        """Run all stress scenarios and compare results."""
        from src.risk.expected_loss import ExpectedLossCalculator
        el_calc = ExpectedLossCalculator()

        results = {}

        # Baseline
        X_base = feature_table[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        base_pd = model.predict_proba(X_base)[:, 1]
        base_el = el_calc.compute_expected_loss(
            pd.Series(base_pd), credit_limits, current_balances
        )
        results["baseline"] = {
            "avg_pd": round(float(base_pd.mean()), 4),
            "total_expected_loss": round(float(base_el["expected_loss"].sum()), 2),
            "default_rate": round(float((base_pd > 0.5).mean()), 4),
        }

        # Stress scenarios
        for scenario_name in self.scenarios:
            stressed = self.apply_stress(feature_table, scenario_name)
            X_stressed = stressed[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
            stressed_pd = model.predict_proba(X_stressed)[:, 1]
            stressed_el = el_calc.compute_expected_loss(
                pd.Series(stressed_pd), credit_limits, current_balances
            )

            results[scenario_name] = {
                "avg_pd": round(float(stressed_pd.mean()), 4),
                "total_expected_loss": round(float(stressed_el["expected_loss"].sum()), 2),
                "default_rate": round(float((stressed_pd > 0.5).mean()), 4),
                "el_increase_pct": round(
                    (stressed_el["expected_loss"].sum() / base_el["expected_loss"].sum() - 1) * 100, 2
                ),
                "pd_increase_pct": round(
                    (stressed_pd.mean() / base_pd.mean() - 1) * 100, 2
                ),
            }
            logger.info(
                f"Scenario '{scenario_name}': "
                f"EL increase = {results[scenario_name]['el_increase_pct']:.1f}%, "
                f"PD increase = {results[scenario_name]['pd_increase_pct']:.1f}%"
            )

        return results
