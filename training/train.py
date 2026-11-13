"""Train a model and log everything (incl. dataset provenance) to MLflow."""
import argparse
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=5)
parser.add_argument("--dataset-version", default="v1")
parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
parser.add_argument("--experiment", default="mlops-copilot-v3")
args = parser.parse_args()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def dvc_md5(path: str = "data/dataset.csv.dvc") -> str:
    with open(path) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


mlflow.set_tracking_uri(args.tracking_uri)
mlflow.set_experiment(args.experiment)

df = pd.read_csv("data/dataset.csv")
X, y = df.drop(columns=["target"]), df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    mlflow.set_tags({
        "dataset_version": args.dataset_version,
        "dvc_md5": dvc_md5(),
        "git_commit": git_commit(),
    })
    mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    ).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred),
    })
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    print("run_id:", run.info.run_id)