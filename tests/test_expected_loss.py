"""Tests for expected loss calculations."""

import pytest
import pandas as pd
from src.risk.expected_loss import ExpectedLossCalculator


@pytest.fixture
def calculator():
    calc = ExpectedLossCalculator.__new__(ExpectedLossCalculator)
    calc.lgd_mean = 0.45
    calc.lgd_std = 0.1
    calc.ead_factor = 0.75
    return calc


class TestExpectedLoss:
    def test_lgd_range(self, calculator):
        lgd = calculator.compute_lgd(1000)
        assert lgd.min() >= 0.05
        assert lgd.max() <= 0.95
        assert 0.3 < lgd.mean() < 0.6

    def test_ead_calculation(self, calculator):
        limits = pd.Series([10000, 20000, 5000])
        balances = pd.Series([3000, 15000, 2000])
        ead = calculator.compute_ead(limits, balances)
        assert (ead >= balances).all()
        assert (ead <= limits * 2).all()

    def test_expected_loss_positive(self, calculator):
        pd_vals = pd.Series([0.1, 0.3, 0.05])
        limits = pd.Series([10000, 20000, 5000])
        balances = pd.Series([3000, 15000, 2000])
        result = calculator.compute_expected_loss(pd_vals, limits, balances)
        assert (result["expected_loss"] >= 0).all()
        assert "PD" in result.columns
        assert "LGD" in result.columns
        assert "EAD" in result.columns

    def test_zero_pd_low_loss(self, calculator):
        pd_vals = pd.Series([0.0, 0.0])
        limits = pd.Series([10000, 20000])
        balances = pd.Series([5000, 10000])
        result = calculator.compute_expected_loss(pd_vals, limits, balances)
        assert result["expected_loss"].sum() == 0

    def test_portfolio_summary(self, calculator):
        pd_vals = pd.Series([0.1, 0.2, 0.05, 0.3])
        limits = pd.Series([10000, 20000, 5000, 15000])
        balances = pd.Series([3000, 15000, 2000, 10000])
        el_results = calculator.compute_expected_loss(pd_vals, limits, balances)
        summary = calculator.portfolio_summary(el_results)
        assert "total_expected_loss" in summary
        assert "loss_rate" in summary
        assert summary["n_customers"] == 4
        assert summary["loss_rate"] >= 0
