# Production Medical RAG & Evaluation Pipeline (v1.0.0)

A production-ready, enterprise-grade **Medical Retrieval-Augmented Generation (RAG) Architecture and Evaluation System** built with Python 3.11, FastAPI, Qdrant, Weaviate, LangGraph, SentenceTransformers, Groq API, and MLflow.

---

## 🌟 Key Features & System Capabilities

- **Modular Architecture**: Layered design decoupling Ingestion, Chunking, Embedding, Vector Stores, Retrievers, LLMs, and Evaluation.
- **Hybrid Search & Fusion**: Dense vector similarity + BM25 Okapi lexical matching merged via weighted Reciprocal Rank Fusion (RRF).
- **GPU Accelerated Reranking**: CUDA-accelerated cross-encoder reranker for context precision optimization.
- **Dual RAG Orchestration**:
  - **Linear RAG Pipeline**: Low-latency ($\sim 400\text{ms}$) direct synthesis flow.
  - **Agentic RAG Graph (LangGraph)**: Self-correcting state graph with document relevance grading and adaptive query rewriting loops.
- **3-Layer Evaluation Harness**:
  - **Layer 1 (IR Metrics)**: $MRR@K$, $NDCG@K$, $Precision@K$, $Recall@K$ via `ranx`.
  - **Layer 2 (LLM Quality)**: $Faithfulness$ (hallucination detection) and $Answer\ Relevance$.
  - **Layer 3 (System)**: End-to-end latency, index build time, token counts, cost tracking.
- **MLflow Experiment Sweep Runner (M8)**: Automated Cartesian product matrix benchmarking across vector DBs, embedding models, chunking strategies, and orchestration types with MLflow run caching and automated Markdown report generation (`reports/comparison_report.md`).
- **Production REST API (FastAPI)**: Serves clinical queries (`/query`), triggers CI/CD evaluation sweeps (`/evaluate`), and logs physician ratings (`/feedback`).

---

## 🏗️ System Architecture

![System Architecture](./assets/architectural_diagram.jpg)


---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- NVIDIA GPU (Optional, CUDA supported for embeddings/reranking)

### 1. Environment Setup & Installation

```bash
# Clone the repository
cd rag_evaluation_pipeline

# Create python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install --upgrade pip hatchling
pip install -e .
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Set GROQ_API_KEY and MLFLOW_TRACKING_URI in .env
```

### 3. Start Local Microservices (Qdrant, Weaviate, MLflow)

```bash
docker-compose up -d
```

Services will be running at:
- **Qdrant Vector Store**: `http://localhost:6333`
- **Weaviate Vector Store**: `http://localhost:8080`
- **MLflow Tracking UI**: `http://localhost:5000`

---

## ⚡ Running the FastAPI Server

Start the REST API server:

```bash
uvicorn rag_eval.api.main:app --reload --port 8000
```

Access interactive Swagger documentation at:
👉 **`http://localhost:8000/docs`**

### REST API Endpoints:
- `POST /query`: Execute medical RAG query (Linear or Agentic).
- `POST /evaluate`: Trigger evaluation suite and log run to MLflow.
- `POST /feedback`: Log physician ratings (1-5 stars) to MLflow.

---

## 🧪 Running Matrix Experiment Sweeps (M8)

Execute automated matrix expansion and evaluation sweeps:

```bash
python scripts/test_pipeline_m0_to_m8.py
```

Generated production recommendation report will be saved to:
📄 **`reports/comparison_report.md`**

---

## 🛡️ Testing & Quality Gates

Run all quality checks and unit tests:

```bash
make check
```

Or run individual commands:

```bash
# Code linting & formatting
ruff check .
ruff format --check .

# Type checking
mypy src/

# Security audit
bandit -r src/

# Run unit & integration tests
pytest --cov=src/rag_eval tests/
```

---

## 🐳 Docker Deployment

Build and run the production container:

```bash
docker build -t rag-eval:1.0.0 .
docker run -p 8000:8000 --env-file .env rag-eval:1.0.0
```

---

## 📄 License & Version
- **Version**: `1.0.0`
- **License**: MIT
