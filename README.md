# MLOps Copilot

An AI agent that answers questions about an ML system by calling real tools: **DVC** (dataset provenance), **MLflow** (experiments and registry), **Apache Airflow 3** (training pipeline) and **Kubernetes** (inference service). Built for Assignment 03, MLOps & Production AI Engineering, Fall 2026.

The agent does not answer operational questions from its own knowledge. Every fact in an answer comes from a tool result, and tool failures are reported instead of guessed.

## Architecture

See [architecture.md](architecture.md). In short: `FastAPI /chat` -> `orchestrator` (LLM + tool routing + permissions + logging) -> 4 tool modules -> evidence-based answer (known facts, uncertainties, recommended actions).

## Requirements

Windows with WSL2 (Ubuntu 22.04), Docker Desktop, Python 3.12, `kind` and `kubectl`.

```bash
git clone https://github.com/Hhumna/mlops-copilot.git
cd mlops-copilot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the LLM settings
```

`.env` needs `LLM_API_KEY` (or comma-separated `LLM_API_KEYS` for fallback keys), `LLM_BASE_URL` and `LLM_MODEL` (any OpenAI-compatible provider; this project used Groq). Never commit `.env`.

## Dataset (DVC)

The dataset is tracked with DVC and stored in a local DVC remote, so `dvc pull` will not work on another machine. Recreate it instead:

```bash
python training/make_dataset.py --version v2   # v1 = 569 rows, v2 = 469 rows
dvc status
```

Reproduce a past version: `git checkout data-v1 -- data/dataset.csv.dvc && dvc checkout data/dataset.csv.dvc`.

## Launching the services

**MLflow** (host, port 5000):

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts --serve-artifacts --allowed-hosts "*"
python training/train.py --n-estimators 100 --max-depth 5 --dataset-version v2
```

**Airflow 3** (Docker Compose, port 8080, login `airflow` / `airflow`):

```bash
cd airflow
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up airflow-init
docker compose up -d
```

The `training_pipeline` DAG (validate -> preprocess -> train -> evaluate/log to MLflow -> quality gate) is mounted from `dags/`. Params: `n_estimators`, `max_depth`, `dataset_version`, `min_f1`.

**Kubernetes** (kind, namespace `mlops-app`):

```bash
kind create cluster --name mlops
kubectl create namespace mlops-app
docker build -t mlops-inference:1.0.0 .
kind load docker-image mlops-inference:1.0.0 --name mlops
kubectl apply -f kubernetes/
```

The image bakes in `api/model.joblib`. If it is missing, create it from the dataset with `RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)` and save `{"model", "features", "dataset_version"}` with joblib.

**Agent API** (port 8100):

```bash
uvicorn agent.server:app --port 8100
curl -s localhost:8100/chat -H "Content-Type: application/json" \
  -d '{"message":"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."}'
```

CLI alternative: `python -m agent.main "your question"`.

## Safety

- DVC: only whitelisted read-only commands, refs and paths validated.
- Airflow: only the approved DAG can be triggered, only known `conf` keys, and only with user confirmation.
- Kubernetes: operations limited to namespace `mlops-app` and deployment `inference`; changes need `"confirm_actions": true` in the request. The model cannot confirm on the user's behalf.
- Retraining is never started automatically. It is recommended only with evidence (failed quality gate, freshness policy, or degradation results).
- Structured JSON logs (request id, tool, status, duration); secrets are never logged.

## Tests and CI

```bash
ruff check .
python -m pytest -q
```

All external services are mocked, so tests run without MLflow, Airflow, Kubernetes or an LLM. GitHub Actions (`.github/workflows/ci.yml`) runs lint and tests on pull requests.

## Evidence

Tested agent conversations (real tool output) are in `evidence/`. Screenshots are in `screenshots/`.

## Known limitations

- The model registry is empty; no model is promoted automatically.
- The inference image bakes in its model file instead of loading it from the registry.
- The agent cannot read deployment env vars or the ConfigMap, so it can miss configuration-level root causes.
- Free-tier LLM rate limits can interrupt multi-tool requests; keys rotate automatically when `LLM_API_KEYS` is set.
- Recommended actions written by the LLM can go beyond tool evidence; only the "Known facts" are evidence-backed.