#!/usr/bin/env python3
"""
End-to-end pipeline runner.

Executes the full pipeline: data generation → feature engineering →
model training → risk calculations → survival analysis → drift simulation.
Saves results to reports/pipeline_results.json for README embedding.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --quick  # smaller dataset for CI
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(quick: bool = False) -> dict:
    start = time.time()
    results = {"timestamp": datetime.utcnow().isoformat(), "stages": {}}

    # ── 1. Data Generation ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1: Synthetic Data Generation")
    if quick:
        import yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg["data_generation"]["n_customers"] = 1000
        cfg["data_generation"]["n_months"] = 12
        with open("configs/config.yaml", "w") as f:
            yaml.dump(cfg, f)
        logger.info("Quick mode: 1,000 customers, 12 months")

    from src.data_generation.generate_synthetic_data import SyntheticBankDataGenerator
    t0 = time.time()
    gen = SyntheticBankDataGenerator()
    datasets = gen.generate_all()
    results["stages"]["data_generation"] = {
        "n_customers": len(datasets["customers"]),
        "n_months": gen.n_months,
        "n_transactions": len(datasets["transactions"]),
        "default_rate": round(float(datasets["defaults"]["is_default"].mean()), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    logger.info(f"  Customers: {len(datasets['customers']):,}")
    logger.info(f"  Transactions: {len(datasets['transactions']):,}")
    logger.info(f"  Default rate: {datasets['defaults']['is_default'].mean():.2%}")

    # ── 2. Feature Engineering ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2: Feature Engineering")
    t0 = time.time()
    from src.pipelines.feature_pipeline import CreditFeatureEngineer
    engineer = CreditFeatureEngineer()
    feature_table = engineer.build_feature_table(datasets)

    n_features = feature_table.shape[1]
    results["stages"]["feature_engineering"] = {
        "n_features": n_features,
        "n_customers": len(feature_table),
        "elapsed_s": round(time.time() - t0, 1),
    }
    logger.info(f"  Feature table: {feature_table.shape}")

    # ── 3. Model Training ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3: Model Training")
    t0 = time.time()
    from src.models.train import CreditRiskModelTrainer
    trainer = CreditRiskModelTrainer()
    training_output = trainer.train_all(feature_table)

    model_comparison = []
    for model_name, metrics in training_output["results"].items():
        model_comparison.append({
            "model": model_name,
            "roc_auc": round(metrics["roc_auc"], 4),
            "pr_auc": round(metrics["pr_auc"], 4),
            "f1": round(metrics["f1"], 4),
            "ks_statistic": round(metrics["ks_statistic"], 4),
            "brier_score": round(metrics["brier_score"], 4),
        })

    results["stages"]["model_training"] = {
        "best_model": training_output["best_model"],
        "best_auc": round(training_output["results"][training_output["best_model"]]["roc_auc"], 4),
        "model_comparison": model_comparison,
        "walk_forward_auc": round(training_output["walk_forward"]["walk_forward_mean_auc"], 4),
        "walk_forward_std": round(training_output["walk_forward"]["walk_forward_std_auc"], 4),
        "n_features_used": len(training_output["feature_names"]),
        "elapsed_s": round(time.time() - t0, 1),
    }
    logger.info(f"  Best model: {training_output['best_model']} "
                f"(AUC={results['stages']['model_training']['best_auc']:.4f})")

    # ── 4. Risk Calculations ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4: Expected Loss & Stress Testing")
    t0 = time.time()
    from src.risk.expected_loss import ExpectedLossCalculator
    from src.risk.stress_testing import StressTester

    best_model = training_output["models"][training_output["best_model"]]
    X_test = training_output["test_data"]["X_test"]

    calc = ExpectedLossCalculator()
    pd_vals = pd.Series(best_model.predict_proba(X_test)[:, 1])
    cards = datasets["credit_cards"]
    stmts = datasets["statements"]
    latest_balance = stmts.sort_values("statement_date").groupby("customer_id")["statement_balance"].last()

    # Align sizes
    n = min(len(pd_vals), len(cards))
    limits = cards["credit_limit"].values[:n]
    balances = np.array([latest_balance.iloc[i] if i < len(latest_balance) else 0 for i in range(n)])
    pd_vals_aligned = pd_vals.values[:n]

    el_results = calc.compute_expected_loss(
        pd.Series(pd_vals_aligned), pd.Series(limits), pd.Series(balances)
    )
    portfolio_summary = calc.portfolio_summary(el_results)

    # Stress testing
    tester = StressTester()
    stress_results = tester.run_all_scenarios(el_results, datasets["customers"].iloc[:n])

    results["stages"]["risk"] = {
        "portfolio_expected_loss": round(float(portfolio_summary["total_expected_loss"]), 2),
        "loss_rate_pct": round(float(portfolio_summary["loss_rate"]) * 100, 3),
        "mean_pd": round(float(pd_vals_aligned.mean()), 4),
        "stress_results": {
            scenario: {
                "loss_increase_pct": round(float(data.get("loss_increase_pct", 0)), 2)
            }
            for scenario, data in stress_results.items()
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    # ── 5. Survival Analysis ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 5: Survival Analysis")
    t0 = time.time()
    try:
        from src.models.survival_model import CreditSurvivalAnalyzer
        survival = CreditSurvivalAnalyzer()
        survival_df = survival.prepare_survival_data(feature_table, n_months=gen.n_months)
        survival.fit(survival_df)
        km_results = survival.fit_kaplan_meier(survival_df)

        cox_c_index = survival.cox_model.concordance_index_
        results["stages"]["survival_analysis"] = {
            "cox_concordance_index": round(float(cox_c_index), 4),
            "km_segments": {
                seg: {
                    "median_survival_months": v.get("median_survival_months"),
                    "n": v.get("n"),
                }
                for seg, v in km_results.items()
                if isinstance(v, dict) and "median_survival_months" in v
            },
            "elapsed_s": round(time.time() - t0, 1),
        }
        logger.info(f"  Cox concordance index: {cox_c_index:.4f}")
    except Exception as e:
        logger.warning(f"Survival analysis skipped: {e}")
        results["stages"]["survival_analysis"] = {"error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────────
    results["total_elapsed_s"] = round(time.time() - start, 1)
    results["status"] = "success"

    os.makedirs("reports", exist_ok=True)
    results_path = "reports/pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {results['total_elapsed_s']}s")
    logger.info(f"Results saved to {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run with smaller dataset")
    args = parser.parse_args()
    results = run_pipeline(quick=args.quick)

    # Print summary table
    mt = results["stages"]["model_training"]
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<25} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'KS':>8}")
    print("-" * 60)
    for row in mt["model_comparison"]:
        star = " \u2605" if row["model"] == mt["best_model"] else ""
        print(f"{row['model']:<25} {row['roc_auc']:>8.4f} {row['pr_auc']:>8.4f} "
              f"{row['f1']:>8.4f} {row['ks_statistic']:>8.4f}{star}")
    print(f"\nWalk-forward CV AUC: {mt['walk_forward_auc']:.4f} \u00b1 {mt['walk_forward_std']:.4f}")

    if "survival_analysis" in results["stages"]:
        sa = results["stages"]["survival_analysis"]
        if "cox_concordance_index" in sa:
            print(f"Cox PH concordance index: {sa['cox_concordance_index']:.4f}")
