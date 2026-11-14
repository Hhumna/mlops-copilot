import json
import logging
import sys

from agent.orchestrator import Agent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("> ")
    r = Agent().run(question)
    print("\n" + r["answer"] + "\n")
    print(json.dumps({k: r[k] for k in ("request_id", "status", "tools_used")}, indent=1))