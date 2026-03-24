"""Health check helpers for liveness and readiness endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from services.db import check_database_ready
from services.vector_store import check_qdrant_ready


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_liveness_payload() -> dict:
    """Return a lightweight liveness payload."""
    return {
        "status": "ok",
        "timestamp": _utc_now(),
        "checks": {
            "application": {
                "status": "ok",
            }
        },
    }


async def build_readiness_payload() -> tuple[dict, bool]:
    """Return readiness details and whether the service is ready."""
    db_ok, db_detail = await check_database_ready()
    qdrant_ok, qdrant_detail = await check_qdrant_ready()

    ready = db_ok and qdrant_ok
    payload = {
        "status": "ok" if ready else "degraded",
        "timestamp": _utc_now(),
        "checks": {
            "database": {
                "status": "ok" if db_ok else "error",
                "detail": db_detail,
            },
            "qdrant": {
                "status": "ok" if qdrant_ok else "error",
                "detail": qdrant_detail,
            },
        },
    }
    return payload, ready
