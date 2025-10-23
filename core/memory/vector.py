"""Vector memory store (stub for future semantic search)."""

from typing import Optional

try:
    import faiss
    import numpy as np

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorMemoryStore:
    """Vector-based semantic memory search (optional, stub for future)."""

    def __init__(self, dimension: int = 384):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.index = None
        self.memories: list[str] = []

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dimension)

    def add_memory(self, text: str, embedding: Optional[list[float]] = None):
        """
        Add a memory with its embedding.

        Args:
            text: Memory text
            embedding: Optional pre-computed embedding
        """
        self.memories.append(text)

        if self.index and embedding:
            vec = np.array([embedding], dtype=np.float32)
            self.index.add(vec)

    def search(self, query_embedding: list[float], k: int = 5) -> list[str]:
        """
        Search for similar memories.

        Args:
            query_embedding: Query vector
            k: Number of results

        Returns:
            List of matching memory texts
        """
        if not self.index or len(self.memories) == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, min(k, len(self.memories)))

        return [self.memories[i] for i in indices[0]]

    def keyword_search(self, query: str, k: int = 5) -> list[str]:
        """
        Simple keyword-based search (fallback).

        Args:
            query: Query text
            k: Number of results

        Returns:
            Matching memories
        """
        query_lower = query.lower()
        matches = [m for m in self.memories if query_lower in m.lower()]
        return matches[:k]
