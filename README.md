# Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

# Backend

```bash
cd backend
cp .env.example .env          # fill in OPENAI_API_KEY
uv sync                       # install dependencies
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

# Frontend

```bash
cd frontend
npm install
npm run dev                   # starts on :5173 (proxies /api → :8000)
```
