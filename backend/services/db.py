"""Async SQLAlchemy session factory."""

import logging
from urllib.parse import urlparse

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config import settings

logger = logging.getLogger(__name__)


async def ensure_database_exists() -> None:
    """Create the target PostgreSQL database if it does not already exist.

    Connects to the default ``postgres`` maintenance database using the same
    credentials as the application, then issues a ``CREATE DATABASE`` statement
    when the target database is absent.
    """
    parsed = urlparse(settings.database_url)
    db_name = parsed.path.lstrip("/")            # e.g. "spatial_notes"
    user = parsed.username
    password = parsed.password
    host = parsed.hostname
    port = parsed.port or 5432

    # Connect to the maintenance database instead of the target one
    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database="postgres",
    )
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        if not exists:
            # CREATE DATABASE cannot run inside a transaction block
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            logger.info("Database '%s' created successfully.", db_name)
        else:
            logger.debug("Database '%s' already exists.", db_name)
    finally:
        await conn.close()


engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency that yields an async DB session."""
    async with async_session() as session:
        yield session
