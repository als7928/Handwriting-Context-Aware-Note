# Backend — Spatial-Aware Note Agent

FastAPI-based backend server. Extracts text and handwritten annotations from PDF documents and answers queries through a LangGraph RAG pipeline.

## Tech Stack

| Component | Role |
|-----------|------|
| FastAPI + Uvicorn | HTTP API server |
| SQLAlchemy (asyncpg) | Async PostgreSQL ORM |
| Qdrant | Vector similarity search |
| LangGraph | RAG pipeline orchestration |
| DSPy | Query rewriting (LLM prompt optimization) |
| PyMuPDF | PDF parsing + annotation extraction |
| EasyOCR | Handwritten text recognition |
| OpenAI | Embeddings (text-embedding-3-small) + LLM (gpt-4o-mini) |

## Directory Structure

```
backend/
├── main.py              FastAPI app entry point, lifespan (DB & Qdrant init)
├── config.py            Environment config (Pydantic Settings)
├── logging_config.py    Logging setup
├── agents/
│   ├── graph.py         LangGraph node connection graph
│   ├── query_rewrite.py DSPy – splits query into semantic_query + marker_filter
│   ├── retriever.py     Qdrant hybrid search (vector + metadata filter)
│   ├── reranker.py      Result reranking
│   ├── synthesis.py     GPT-based response synthesis
│   └── state.py         LangGraph shared state schema
├── api/
│   ├── documents.py     POST /documents/upload, GET /documents/, DELETE
│   └── chat.py          POST /chat/
├── models/
│   ├── database.py      SQLAlchemy ORM (Document, SpatialChunk)
│   └── schemas.py       Pydantic request/response schemas
└── services/
    ├── db.py            AsyncSession factory, ensure_database_exists()
    ├── embedding.py     OpenAI embedding calls
    ├── ocr.py           EasyOCR wrapper
    ├── spatial_chunker.py PDF → SpatialChunk conversion pipeline
    └── vector_store.py  Qdrant upsert / hybrid_search (with MMR)
```

## Local Development Setup

### Prerequisites
- Python 3.11.9
- [uv](https://docs.astral.sh/uv/) installed
- PostgreSQL running (or via Docker)
- Qdrant running

### 1. Configure Environment

```bash
cp .env.example .env
```

Required `.env` fields:

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:1234@localhost:5432/spatial_notes
QDRANT_HOST=localhost
QDRANT_PORT=6333
OPENAI_API_KEY=sk-...          # required – enter your OpenAI API key
```

> The `spatial_notes` database is automatically created on startup if it does not exist.

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run the Server

```bash
# Development mode (hot-reload)
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# or
uv run python main.py
```

### 4. Health Check

```bash
curl http://127.0.0.1:8000/api/health
# {"status":"ok"}
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/documents/upload` | Upload a PDF, trigger chunking & embedding |
| `GET` | `/api/documents/` | List uploaded documents |
| `GET` | `/api/documents/{id}/file` | Download the original PDF |
| `DELETE` | `/api/documents/{id}` | Delete document (DB + Qdrant) |
| `POST` | `/api/chat/` | Query answering (RAG pipeline) |
| `GET` | `/api/health` | Server health check |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:1234@localhost:5432/spatial_notes` | PostgreSQL connection URL |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | `spatial_chunks` | Qdrant collection name |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIM` | `1536` | Embedding dimension |
| `LLM_MODEL` | `gpt-4o-mini` | LLM for response generation |
| `UPLOAD_DIR` | `uploads` | PDF storage path (relative) |
| `APP_ENV` | `production` | Set to `development` to force DEBUG log level |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_FILE` | `logs/app.log` | Log file path |

## Running with Docker

Using the root `docker-compose.yml` is recommended.  
To build the backend image standalone:

```bash
docker build -t spatial-note-backend .
docker run -p 8000:8000 --env-file .env spatial-note-backend
```

