"""Inference service: /health, /ready, /version, /predict."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("inference")

APP_VERSION = os.getenv("APP_VERSION", "dev")
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "model.joblib"))

app = FastAPI(title="mlops-copilot inference")
_bundle: dict | None = None


def _load() -> dict:
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    return _bundle


class PredictRequest(BaseModel):
    features: list[float]


@app.on_event("startup")
def startup() -> None:
    try:
        _load()
    except Exception:  # readiness probe will report not ready
        log.exception("Model failed to load")


@app.get("/health")
def health() -> dict:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    return {
        "app_version": APP_VERSION,
        "dataset_version": _bundle["dataset_version"] if _bundle else None,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    n = len(_bundle["features"])
    if len(req.features) != n:
        raise HTTPException(status_code=422, detail=f"expected {n} features")
    return {"prediction": int(_bundle["model"].predict([req.features])[0])}