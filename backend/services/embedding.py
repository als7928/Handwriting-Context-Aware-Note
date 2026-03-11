"""OpenAI embedding helper – async, batched."""

from __future__ import annotations

from openai import AsyncOpenAI

from config import settings

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a batch of texts.

    Splits into chunks of 2048 to stay within the OpenAI API limit.
    """
    client = _get_client()
    results: list[list[float]] = []
    batch_size = 2048
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(
            input=batch,
            model=settings.embedding_model,
        )
        results.extend(item.embedding for item in response.data)
    return results


async def embed_single(text: str) -> list[float]:
    """Return embedding vector for a single text."""
    vecs = await embed_texts([text])
    return vecs[0]
