"""Embedding client for the per-account semantic knowledge base.

Talks to the local TEI (text-embeddings-inference) server running bge-small-zh.
Vectors are stored per-account and searched with brute-force cosine (each
account's knowledge base is small, so an ANN index is unnecessary).
"""
from __future__ import annotations

import math

import httpx

from .config import get_settings


def embed_enabled() -> bool:
    return bool(get_settings().embed_url)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    if not settings.embed_url or not texts:
        return []
    url = settings.embed_url.rstrip("/") + "/embed"
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        resp = await client.post(url, json={"inputs": texts})
        resp.raise_for_status()
        data = resp.json()
    return data if isinstance(data, list) else []


async def embed_text(text: str) -> list[float]:
    vectors = await embed_texts([text])
    return vectors[0] if vectors else []


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
