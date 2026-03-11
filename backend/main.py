"""Spatial-Aware Note Agent – FastAPI application entry point."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from config import settings
from api.documents import router as documents_router
from models.database import Base
from services.db import engine
from services.vector_store import ensure_collection


# Initialise logging as early as possible so all module-level loggers pick it up
setup_logging(settings.effective_log_level, settings.log_file)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables & Qdrant collection. Shutdown: dispose engine."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        await ensure_collection()
    except Exception as exc:
        if settings.require_qdrant_on_startup:
            raise
        logger.warning(
            "Qdrant is unavailable at startup (%s). Continuing without vector store initialization.",
            exc,
        )
    yield
    await engine.dispose()


app = FastAPI(
    title="Spatial-Aware Note Agent",
    description="Gemini-like AI service for querying handwritten-annotated PDFs.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS – allow the Vue.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/api/health")
async def health_check():
    """Simple readiness probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
