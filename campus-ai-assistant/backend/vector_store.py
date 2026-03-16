"""
Vector store abstraction with Endee backend + numpy fallback.

If the Endee server isn't running, we fall back to a simple in-memory
cosine similarity store so the app still works for demos.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


class VectorStore(ABC):
    """Base interface for vector stores."""

    @abstractmethod
    def connect(self, dimension=384):
        ...

    @abstractmethod
    def upsert(self, items):
        ...

    @abstractmethod
    def query(self, vector, top_k=5):
        ...

    @abstractmethod
    def count(self):
        ...

    @property
    @abstractmethod
    def store_type(self):
        ...


class EndeeVectorStore(VectorStore):
    """Connects to the Endee vector database via the Python SDK."""

    def __init__(self, base_url="http://localhost:8080/api/v1"):
        self.base_url = base_url
        self.index_name = "campus_docs"
        self.client = None
        self.index = None
        self._connected = False
        self._count = 0

    def connect(self, dimension=384):
        try:
            from endee import Endee, Precision

            self.client = Endee()
            self.client.set_base_url(self.base_url)

            try:
                self.client.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    space_type="cosine",
                    precision=Precision.INT8,
                )
                log.info("Created index '%s' (dim=%d)", self.index_name, dimension)
            except Exception:
                log.info("Index '%s' already exists, reusing", self.index_name)

            self.index = self.client.get_index(name=self.index_name)
            self._connected = True
            log.info("Connected to Endee at %s", self.base_url)
            return True
        except Exception as e:
            log.warning("Endee not available (%s): %s", self.base_url, e)
            return False

    @property
    def store_type(self):
        return "endee"

    def count(self):
        return self._count

    def upsert(self, items):
        if not self._connected:
            raise ConnectionError("Not connected to Endee")

        # batch upsert in chunks of 100
        for i in range(0, len(items), 100):
            batch = items[i:i + 100]
            try:
                self.index.upsert(batch)
            except Exception as e:
                log.error("Upsert failed at batch %d: %s", i, e)
                raise
        self._count += len(items)

    def query(self, vector, top_k=5):
        if not self._connected:
            raise ConnectionError("Not connected")

        raw = self.index.query(vector=vector, top_k=top_k)
        results = []
        for r in raw:
            if isinstance(r, dict):
                results.append({
                    "id": r.get("id", ""),
                    "similarity": float(r.get("similarity", r.get("score", 0))),
                    "meta": r.get("meta", {}),
                })
            else:
                results.append({
                    "id": getattr(r, "id", ""),
                    "similarity": float(getattr(r, "similarity",
                                                getattr(r, "score", 0))),
                    "meta": getattr(r, "meta", {}),
                })
        return results


class InMemoryVectorStore(VectorStore):
    """Fallback store using numpy. No Endee server needed."""

    def __init__(self):
        self._id_map = {}   # id -> index
        self._vectors = []
        self._metadata = []
        self._ids = []

    def connect(self, dimension=384):
        log.info("Using in-memory vector store (fallback mode)")
        return True

    @property
    def store_type(self):
        return "in_memory"

    def count(self):
        return len(self._ids)

    def upsert(self, items):
        for item in items:
            vec = np.array(item["vector"], dtype=np.float32)
            item_id = item["id"]
            meta = item.get("meta", {})

            if item_id in self._id_map:
                idx = self._id_map[item_id]
                self._vectors[idx] = vec
                self._metadata[idx] = meta
            else:
                self._id_map[item_id] = len(self._ids)
                self._ids.append(item_id)
                self._vectors.append(vec)
                self._metadata.append(meta)

    def query(self, vector, top_k=5):
        if not self._vectors:
            return []

        q = np.array(vector, dtype=np.float32)
        mat = np.stack(self._vectors)
        sims = mat @ q  # cosine sim (vectors are normalized)

        k = min(top_k, len(self._vectors))
        if k < len(sims):
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        else:
            top_idx = np.argsort(sims)[::-1]

        return [
            {"id": self._ids[i], "similarity": float(sims[i]),
             "meta": self._metadata[i]}
            for i in top_idx
        ]


def create_vector_store(endee_url="http://localhost:8080/api/v1", dimension=384):
    """Try Endee first, fall back to in-memory."""
    store = EndeeVectorStore(base_url=endee_url)
    if store.connect(dimension=dimension):
        return store

    log.warning("Falling back to in-memory vector store")
    fallback = InMemoryVectorStore()
    fallback.connect(dimension=dimension)
    return fallback
