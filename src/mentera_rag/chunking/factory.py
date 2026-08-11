from typing import Any

from mentera_rag.chunking.base import BaseChunker
from mentera_rag.chunking.recursive import RecursiveCharacterChunker


class ChunkerFactory:
    """Factory class to dynamically instantiate chunkers based on strategy name."""

    _registry: dict[str, type[BaseChunker]] = {
        "recursive": RecursiveCharacterChunker,
    }

    @classmethod
    def register(cls, name: str, chunker_cls: type[BaseChunker]) -> None:
        """Allows registering custom external chunking strategies at runtime."""
        cls._registry[name.lower()] = chunker_cls

    @classmethod
    def get_chunker(cls, name: str, **kwargs: Any) -> BaseChunker:
        """Instantiates and returns a chunker strategy by name."""
        key = name.lower()
        if key not in cls._registry:
            valid_keys = list(cls._registry.keys())
            raise ValueError(
                f"Unknown Chunking strategy '{name}'. Available strategies: {valid_keys}"
            )

        return cls._registry[key](**kwargs)
