# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11.9-slim

# System libraries required by easyocr / opencv / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python dependencies before copying the rest of the source so that
# this layer is cached as long as pyproject.toml / uv.lock don't change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source
COPY . .

# Ensure runtime directories exist
RUN mkdir -p uploads logs

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
