"""
Expected Loss Calculator
PD × LGD × EAD computation for portfolio and individual level.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpectedLossCalculator:
    """Computes Expected Loss = PD × LGD × EAD."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.lgd_mean = cfg["risk"]["lgd_mean"]
        self.lgd_std = cfg["risk"]["lgd_std"]
        self.ead_factor = cfg["risk"]["ead_utilization_factor"]

    def compute_lgd(self, n: int) -> np.ndarray:
        """Generate Loss Given Default values (Beta distribution)."""
        alpha = self.lgd_mean * 10
        beta = (1 - self.lgd_mean) * 10
        lgd = np.random.beta(alpha, beta, n).clip(0.05, 0.95)
        return lgd

    def compute_ead(self, credit_limits: pd.Series, current_balances: pd.Series) -> pd.Series:
        """Compute Exposure at Default."""
        unused = credit_limits - current_balances
        ead = current_balances + unused * self.ead_factor
        return ead.clip(0)

    def compute_expected_loss(
        self, pd_values: pd.Series, credit_limits: pd.Series,
        current_balances: pd.Series
    ) -> pd.DataFrame:
        """Calculate Expected Loss for each customer."""
        n = len(pd_values)
        lgd = self.compute_lgd(n)
        ead = self.compute_ead(credit_limits, current_balances)
        el = pd_values.values * lgd * ead.values

        result = pd.DataFrame({
            "PD": pd_values.values.round(4),
            "LGD": lgd.round(4),
            "EAD": ead.values.round(2),
            "expected_loss": el.round(2),
        })

        logger.info(f"Portfolio Expected Loss: ${result['expected_loss'].sum():,.2f}")
        logger.info(f"Average PD: {result['PD'].mean():.2%}")
        logger.info(f"Average LGD: {result['LGD'].mean():.2%}")
        logger.info(f"Average EAD: ${result['EAD'].mean():,.2f}")
        return result

    def portfolio_summary(self, el_results: pd.DataFrame) -> Dict:
        """Summarize portfolio-level expected loss."""
        return {
            "total_expected_loss": round(el_results["expected_loss"].sum(), 2),
            "mean_expected_loss": round(el_results["expected_loss"].mean(), 2),
            "median_expected_loss": round(el_results["expected_loss"].median(), 2),
            "max_expected_loss": round(el_results["expected_loss"].max(), 2),
            "total_ead": round(el_results["EAD"].sum(), 2),
            "loss_rate": round(
                el_results["expected_loss"].sum() / el_results["EAD"].sum(), 4
            ),
            "n_customers": len(el_results),
        }
