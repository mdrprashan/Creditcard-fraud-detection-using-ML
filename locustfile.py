"""
FraudShield — Locust Load Test
Course: ICT946 Capstone | Student: Prashan Manandhar
Run: locust -f locustfile.py
"""
from locust import HttpUser, task, between

class FraudShieldAPIUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://127.0.0.1:8000"

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(2)
    def model_info(self):
        self.client.get("/model-info")

    @task(2)
    def sample_input(self):
        self.client.get("/sample-input")

    @task(5)
    def predict(self):
        self.client.post("/predict", json={"features": [0.0] * 43})

    @task(1)
    def home(self):
        self.client.get("/")

    @task(1)
    def demo_fraud(self):
        self.client.get("/demo-fraud")