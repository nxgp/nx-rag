"""
Disk Memoization Caching for Embeddings.

Prevents redundant token costs and compute time by caching computed vector arrays
on disk keyed by SHA-256 hashes of (model_name, text, dimension).
"""

import hashlib
import json
from pathlib import Path


class EmbeddingCache:
    """
    Simple file-based JSON key-value cache for vector embeddings.
    """

    def __init__(self, cache_dir: Path = Path("data/cache/embeddings")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, model_name: str, text: str, dimension: int) -> str:
        """Generate a unique SHA-256 key for a specific text and model configuration."""
        raw_string = f"{model_name}:{dimension}:{text}"
        return hashlib.sha256(raw_string.encode("utf-8")).hexdigest()

    def get(self, model_name: str, text: str, dimension: int) -> list[float] | None:
        """Retrieve cached vector if present on disk."""
        key = self._compute_key(model_name, text, dimension)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
                vector: list[float] | None = data.get("vector")
                return vector
        return None

    def set(self, model_name: str, text: str, dimension: int, vector: list[float]) -> None:
        """Save computed vector array to disk cache."""
        key = self._compute_key(model_name, text, dimension)
        cache_file = self.cache_dir / f"{key}.json"

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"model": model_name, "dim": dimension, "vector": vector}, f)
