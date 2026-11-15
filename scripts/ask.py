"""Ask the running agent a question. Usage: python scripts/ask.py "question" [--confirm] [--save name]"""
import argparse
import json
import urllib.request
from pathlib import Path


def evidence_name(value: str) -> str:
    if not value or any(not (char.isalnum() or char in "_-") for char in value):
        raise argparse.ArgumentTypeError(
            "save name may contain only letters, numbers, underscores, and hyphens"
        )
    return value


p = argparse.ArgumentParser()
p.add_argument("message")
p.add_argument("--confirm", action="store_true")
p.add_argument("--save", type=evidence_name)
a = p.parse_args()

body = json.dumps(
    {"message": a.message, "confirm_actions": a.confirm}
).encode()
req = urllib.request.Request(
    "http://localhost:8100/chat",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=180) as response:
    d = json.load(response)

print(d["answer"], "\n")
print("tools:", [(t["tool"], t["status"]) for t in d["tools_used"]])
print("status:", d["status"], "| request_id:", d["request_id"])
if a.save:
    evidence_path = Path("evidence") / f"{a.save}.json"
    with evidence_path.open("w", encoding="utf-8") as evidence_file:
        json.dump(d, evidence_file, indent=2)
