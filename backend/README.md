## Run Backend (Uvicorn)

1. Install dependencies

```bash
uv sync
```

2. Start FastAPI with Uvicorn (development)

```bash
uv run python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

3. Health check

```bash
curl http://127.0.0.1:8000/api/health
```

## Notes

- App entrypoint is `main:app`.
- `main.py` also supports direct execution (`python main.py`), but recommended command is `uvicorn main:app`.
