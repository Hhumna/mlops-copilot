"""Shared helpers for all agent tools."""
from __future__ import annotations

import functools
import subprocess


def tool(fn):
    """Return structured {status, data|error} and never raise."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return {"status": "ok", "data": fn(*args, **kwargs)}
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"{fn.__name__} timed out"}
        except PermissionError as e:
            return {"status": "denied", "error": str(e)}
        except (ValueError, FileNotFoundError, KeyError) as e:
            return {"status": "invalid", "error": str(e)}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    return wrapper