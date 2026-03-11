"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the Spatial-Aware Note Agent backend."""

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:1234@localhost:5432/spatial_notes"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "spatial_chunks"
    require_qdrant_on_startup: bool = False

    # OpenAI / LLM
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_model: str = "gpt-4o-mini"

    # File storage
    upload_dir: str = "uploads"

    # Logging
    log_level: str = "INFO"   # Overridden to DEBUG when app_env=development
    log_file: str = "logs/app.log"

    # Runtime environment: development | production
    app_env: str = "production"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def effective_log_level(self) -> str:
        """Return DEBUG in development mode regardless of LOG_LEVEL setting."""
        if self.app_env.lower() == "development":
            return "DEBUG"
        return self.log_level.upper()


settings = Settings()
