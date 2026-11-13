"""Create the versioned dataset. Usage: python training/make_dataset.py [--version v1|v2]"""
import argparse

from sklearn.datasets import load_breast_cancer

parser = argparse.ArgumentParser()
parser.add_argument("--version", default="v1", choices=["v1", "v2"])
args = parser.parse_args()

df = load_breast_cancer(as_frame=True).frame
if args.version == "v2":
    # v2 = more data variety (drop 100 rows) so DVC sees a real change
    df = df.iloc[:-100]
df.to_csv("data/dataset.csv", index=False)
print(f"wrote data/dataset.csv ({len(df)} rows) as {args.version}")
