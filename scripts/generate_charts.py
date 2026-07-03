"""
Generate result charts for the README.
Trains the three models on the feature table and produces:
ROC curves, model comparison, stress test impact, feature importance, lift chart.

Usage:
    python scripts/generate_charts.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier

OUT = "docs/images"
os.makedirs(OUT, exist_ok=True)

COLORS = {"Logistic Regression": "#2563eb", "Random Forest": "#16a34a", "XGBoost": "#dc2626"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_data():
    ft = pd.read_csv("data/processed/feature_table.csv")
    drop = ["customer_id", "is_default", "default_probability", "risk_segment", "composite_risk_score"]
    X = ft.drop(columns=[c for c in drop if c in ft.columns]).select_dtypes(include=[np.number])
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    y = ft["is_default"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_models(X_train, y_train):
    models = {}
    models["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1, random_state=42)),
    ]).fit(X_train, y_train)

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(X_train, y_train)

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, random_state=42, eval_metric="auc"
    ).fit(X_train, y_train)
    return models


def plot_roc(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=COLORS[name], lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Probability of Default Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/roc_curves.png", dpi=150)
    plt.close(fig)
    print("saved roc_curves.png")


def plot_model_comparison(models, X_test, y_test):
    rows = []
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        rows.append({"Model": name, "ROC-AUC": roc_auc_score(y_test, prob),
                     "KS Statistic": float(np.max(tpr - fpr))})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["ROC-AUC"], w, label="ROC-AUC", color="#2563eb")
    ax.bar(x + w / 2, df["KS Statistic"], w, label="KS Statistic", color="#f59e0b")
    for i, row in df.iterrows():
        ax.text(i - w / 2, row["ROC-AUC"] + 0.01, f"{row['ROC-AUC']:.3f}", ha="center", fontsize=10)
        ax.text(i + w / 2, row["KS Statistic"] + 0.01, f"{row['KS Statistic']:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison — Discrimination Power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison.png", dpi=150)
    plt.close(fig)
    print("saved model_comparison.png")


def plot_stress_test():
    with open("reports/stress_test_results.json") as f:
        stress = json.load(f)
    scenarios = ["baseline", "mild", "moderate", "severe"]
    el = [stress[s]["total_expected_loss"] / 1e6 for s in scenarios]
    colors = ["#16a34a", "#f59e0b", "#ea580c", "#7c2d12"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([s.capitalize() for s in scenarios], el, color=colors)
    for bar, s in zip(bars, scenarios):
        label = f"${bar.get_height():.1f}M"
        if s != "baseline":
            label += f"\n(+{stress[s]['el_increase_pct']:.0f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                label, ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Portfolio Expected Loss ($M)")
    ax.set_title("Stress Testing — Portfolio Expected Loss by Scenario")
    ax.set_ylim(0, max(el) * 1.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stress_test.png", dpi=150)
    plt.close(fig)
    print("saved stress_test.png")


def plot_feature_importance(models, X_test):
    xgb = models["XGBoost"]
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance": xgb.feature_importances_,
    }).nlargest(15, "importance").sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"], imp["importance"], color="#2563eb")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("Top 15 Features — XGBoost")
    fig.tight_layout()
    fig.savefig(f"{OUT}/feature_importance.png", dpi=150)
    plt.close(fig)
    print("saved feature_importance.png")


def plot_lift(models, X_test, y_test):
    prob = models["XGBoost"].predict_proba(X_test)[:, 1]
    df = pd.DataFrame({"y": y_test.values, "p": prob})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False) + 1
    lift = df.groupby("decile").agg(rate=("y", "mean")).reset_index()
    lift["lift"] = lift["rate"] / df["y"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(lift["decile"], lift["lift"],
                  color=["#dc2626" if d == 10 else "#94a3b8" for d in lift["decile"]])
    ax.axhline(1, color="k", ls="--", alpha=0.5, label="Baseline (random)")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}x", ha="center", fontsize=9)
    ax.set_xlabel("Score Decile (10 = highest predicted risk)")
    ax.set_ylabel("Lift over Portfolio Default Rate")
    ax.set_title("Lift Chart — XGBoost")
    ax.set_xticks(range(1, 11))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/lift_chart.png", dpi=150)
    plt.close(fig)
    print("saved lift_chart.png")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    models = train_models(X_train, y_train)
    plot_roc(models, X_test, y_test)
    plot_model_comparison(models, X_test, y_test)
    plot_stress_test()
    plot_feature_importance(models, X_test)
    plot_lift(models, X_test, y_test)
    print(f"\nAll charts saved to {OUT}/")
