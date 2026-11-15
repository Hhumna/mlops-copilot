"""Training pipeline: validate -> preprocess -> train -> evaluate/log -> quality gate."""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, task

try:
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:  # older layouts
    from airflow.exceptions import AirflowFailException

log = logging.getLogger("training_pipeline")
PROJECT = Path(os.getenv("PROJECT_DIR", "/opt/airflow/project"))
WORK = PROJECT / ".pipeline_tmp"
DATA = PROJECT / "data" / "dataset.csv"
DVC_FILE = PROJECT / "data" / "dataset.csv.dvc"

DEFAULT_ARGS = {
    "owner": "mlops-copilot",
    "retries": 2,
    "retry_delay": timedelta(seconds=15),
}


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "training"],
    params={
        "n_estimators": Param(100, type="integer", minimum=1, maximum=500),
        "max_depth": Param(5, type="integer", minimum=1, maximum=30),
        "dataset_version": Param("v2", type="string"),
        "min_f1": Param(0.95, type="number", minimum=0, maximum=1),
    },
)
def training_pipeline():
    @task
    def validate_dataset() -> dict:
        import pandas as pd
        import yaml

        if not DATA.exists():
            raise AirflowFailException(f"Dataset missing: {DATA}")
        recorded = yaml.safe_load(DVC_FILE.read_text())["outs"][0]["md5"]
        actual = hashlib.md5(DATA.read_bytes()).hexdigest()
        if actual != recorded:
            raise AirflowFailException(
                f"Dataset md5 {actual} does not match DVC record {recorded}"
            )
        df = pd.read_csv(DATA)
        if "target" not in df.columns:
            raise AirflowFailException("Missing 'target' column")
        if len(df) < 100:
            raise AirflowFailException(f"Too few rows: {len(df)}")
        if df.isnull().any().any():
            raise AirflowFailException("Dataset contains null values")
        log.info("Dataset valid: rows=%d md5=%s", len(df), actual)
        return {"rows": len(df), "md5": actual}

    @task
    def preprocess_data(info: dict, run_id: str | None = None) -> str:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(DATA)
        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["target"]
        )
        out = WORK / (run_id or "manual").replace(":", "_")
        out.mkdir(parents=True, exist_ok=True)
        train.to_csv(out / "train.csv", index=False)
        test.to_csv(out / "test.csv", index=False)
        log.info("Preprocessed: train=%d test=%d -> %s", len(train), len(test), out)
        return str(out)

    @task
    def train_model(split_dir: str, params=None) -> str:
        import joblib
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        train = pd.read_csv(Path(split_dir) / "train.csv")
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            random_state=42,
        ).fit(train.drop(columns=["target"]), train["target"])
        path = Path(split_dir) / "model.joblib"
        joblib.dump(model, path)
        log.info("Model trained and saved to %s", path)
        return str(path)

    @task
    def evaluate_and_log(split_dir: str, model_path: str, info: dict, params=None) -> dict:
        import joblib
        import mlflow
        import mlflow.sklearn
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score

        # relative artifact root of the MLflow server resolves from the project dir
        os.chdir(PROJECT)
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "mlops-copilot-v3"))

        model = joblib.load(model_path)
        test = pd.read_csv(Path(split_dir) / "test.csv")
        pred = model.predict(test.drop(columns=["target"]))
        metrics = {
            "accuracy": float(accuracy_score(test["target"], pred)),
            "f1": float(f1_score(test["target"], pred)),
        }
        with mlflow.start_run(run_name="airflow-training") as run:
            mlflow.set_tags({
                "dataset_version": params["dataset_version"],
                "dvc_md5": info["md5"],
                "git_commit": os.getenv("GIT_COMMIT", "unknown"),
                "trained_by": "airflow",
            })
            mlflow.log_params({
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
            run_id = run.info.run_id
        log.info("Logged MLflow run %s metrics=%s", run_id, metrics)
        return {"mlflow_run_id": run_id, **metrics}

    @task
    def quality_gate(result: dict, params=None) -> dict:
        import mlflow

        threshold = float(params["min_f1"])
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = mlflow.MlflowClient()
        if result["f1"] < threshold:
            client.set_tag(result["mlflow_run_id"], "quality_gate", "failed")
            raise AirflowFailException(
                f"Quality gate FAILED: f1={result['f1']:.4f} < {threshold}. "
                "Model not promoted."
            )
        client.set_tag(result["mlflow_run_id"], "quality_gate", "passed")
        log.info("Quality gate passed: f1=%.4f >= %s", result["f1"], threshold)
        return {"promotable": True, **result}

    info = validate_dataset()
    split = preprocess_data(info)
    model = train_model(split)
    result = evaluate_and_log(split, model, info)
    quality_gate(result)


training_pipeline()