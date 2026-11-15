# Architecture

## Training DAG (dags/training_pipeline.py)

validate dataset (md5 vs DVC record) -> preprocess -> train -> evaluate and log to MLflow -> quality gate.
Validation and the gate fail without retry and block promotion; other tasks retry twice.

## Inference service (api/main.py, kubernetes/)

FastAPI with /health (liveness), /ready (readiness, needs the model loaded), /version, /predict.
Deployment with 2 replicas, probes, CPU/memory requests and limits, ConfigMap, ClusterIP Service.

## Design decisions

- Provenance: every MLflow run stores `dataset_version`, `dvc_md5` and `git_commit` as tags.
- Token limits: only the tool schemas relevant to the question are sent to the LLM; retraining questions get all tools and must check Airflow, the registry and pod health.
- The LLM provider is configurable (OpenAI-compatible API) and keys rotate on rate limits.
