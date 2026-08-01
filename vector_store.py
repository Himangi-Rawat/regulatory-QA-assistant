"""
vector_store.py
---------------
Turns chunks into vectors using Gemini's embedding API, stores them in
a FAISS index for fast similarity search, and keeps a parallel list of
the original chunk text/metadata (FAISS only stores numbers, not text).
"""

import time
import numpy as np
import faiss
from google import genai
from google.genai import types
from chunker import Chunk

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768

_client = None  # set once via configure_gemini()


def configure_gemini(api_key: str):
    """Call this once at the start with your Gemini API key."""
    global _client
    _client = genai.Client(api_key=api_key)


def get_client() -> genai.Client:
    if _client is None:
        raise RuntimeError("Gemini client not configured. Call configure_gemini(api_key) first.")
    return _client


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_delay: float = 0.7) -> np.ndarray:
    """
    task_type = 'RETRIEVAL_DOCUMENT' when embedding chunks to store,
    'RETRIEVAL_QUERY' when embedding the user's question.

    batch_delay defaults to 0.7s between requests to stay under the free
    tier's 100-requests-per-minute embedding limit (60s / 100 = 0.6s min).
    If we still hit a rate limit (e.g. from other traffic sharing the key),
    we back off and retry rather than failing the whole upload.
    """
    client = get_client()
    vectors = []
    for text in texts:
        for attempt in range(8):
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=768,
                    ),
                )
                vectors.append(result.embeddings[0].values)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 8 * (attempt + 1)  # 8s, 16s, 24s... up to ~5.5 min total across 8 tries
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Failed to embed chunk after 8 retries: {text[:50]}...")

        if batch_delay:
            time.sleep(batch_delay)
    return np.array(vectors, dtype="float32")

class VectorStore:
    """Wraps a FAISS index + the chunk metadata list it corresponds to."""

    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], batch_delay: float = 0.0):
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", batch_delay=batch_delay)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return top_k most similar chunks with distance scores (lower = closer)."""
        if self.index.ntotal == 0:
            return []
        query_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")
        distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(dist)))
        return results

    def save(self, path_prefix: str):
        """Save FAISS index + chunk metadata to disk."""
        import pickle
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        with open(f"{path_prefix}.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path_prefix: str) -> "VectorStore":
        import pickle
        store = cls()
        store.index = faiss.read_index(f"{path_prefix}.faiss")
        with open(f"{path_prefix}.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store