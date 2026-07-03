"""
Load Testing — Locust

Tests the credit risk API under concurrent load.
Simulates realistic prediction request patterns.

Usage:
    locust -f locustfile.py --host=http://localhost:8000
    locust -f locustfile.py --host=http://localhost:8000 --headless -u 50 -r 10 --run-time 60s
"""

import random
from locust import HttpUser, between, task


SAMPLE_CUSTOMER = {
    "avg_utilization": 0.45,
    "max_utilization": 0.78,
    "min_utilization": 0.12,
    "std_utilization": 0.18,
    "current_utilization": 0.52,
    "utilization_trend": 0.003,
    "avg_payment_to_balance": 0.65,
    "min_payment_to_balance": 0.10,
    "payment_volatility": 250.0,
    "avg_monthly_payment": 820.0,
    "late_payment_count": 1,
    "total_payment_count": 24,
    "late_payment_ratio": 0.042,
    "spend_mean_3m": 1200.0,
    "spend_std_3m": 180.0,
    "spend_mean_6m": 1150.0,
    "spend_std_6m": 220.0,
    "spend_mean_12m": 1100.0,
    "spend_std_12m": 280.0,
    "spend_trend": 12.5,
    "delinquency_severity": 0.0,
    "delinquency_amount": 0.0,
    "is_delinquent": 0,
    "late_count": 1,
    "avg_transaction_amount": 95.0,
    "max_transaction_amount": 850.0,
    "total_transaction_count": 180,
    "avg_merchant_risk": 0.42,
    "high_risk_txn_ratio": 0.08,
    "international_txn_ratio": 0.03,
    "n_categories": 7,
    "income_to_debt_ratio": 4.2,
    "debt_burden": 0.85,
    "latest_balance": 2800.0,
    "annual_income": 65000.0,
}


def random_customer():
    c = SAMPLE_CUSTOMER.copy()
    c["avg_utilization"] = random.uniform(0.1, 0.9)
    c["late_payment_ratio"] = random.uniform(0, 0.3)
    c["annual_income"] = random.uniform(25000, 200000)
    c["is_delinquent"] = random.choice([0, 0, 0, 1])
    return c


class CreditRiskAPIUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(5)
    def predict_single(self):
        self.client.post("/predict", json=random_customer(), name="/predict")

    @task(2)
    def batch_predict(self):
        batch = {"customers": [random_customer() for _ in range(10)]}
        self.client.post("/batch_predict", json=batch, name="/batch_predict")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(1)
    def model_metrics(self):
        self.client.get("/model_metrics", name="/model_metrics")
