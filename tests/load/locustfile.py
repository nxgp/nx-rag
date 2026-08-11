"""
Locust Performance and Load Testing Script for FastAPI REST Service.

Simulates concurrent user queries, health checks, and evaluation triggers under load.
"""

from locust import HttpUser, between, task


class MedicalRAGUser(HttpUser):
    """
    Simulated user calling Medical RAG REST endpoints.
    """

    wait_time = between(0.5, 2.0)

    @task(3)
    def test_health_check(self):
        """Task 1: Periodic system health check."""
        self.client.get("/health", name="/health")

    @task(5)
    def test_query_linear_rag(self):
        """Task 2: High frequency clinical queries."""
        payload = {
            "query": "Do mitochondria play a role in programmed cell death?",
            "pipeline_type": "linear",
            "top_k": 5,
        }
        self.client.post("/query", json=payload, name="/query [linear]")

    @task(2)
    def test_query_agentic_rag(self):
        """Task 3: Agentic graph queries."""
        payload = {
            "query": "Explain mitochondrial permeability transition in lace plant leaves.",
            "pipeline_type": "agentic",
            "top_k": 5,
        }
        self.client.post("/query", json=payload, name="/query [agentic]")

    @task(1)
    def test_submit_physician_feedback(self):
        """Task 4: Physician rating feedback logging."""
        payload = {
            "run_id": "locust-load-run-001",
            "rating": 5,
            "comments": "Load testing simulated high accuracy rating.",
        }
        self.client.post("/feedback", json=payload, name="/feedback")
