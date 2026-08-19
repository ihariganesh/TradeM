import hashlib
import math
import re
from typing import List


class EmbeddingModel:
    """Embedding model wrapper supporting HuggingFace sentence-transformers and deterministic fallback."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception:
            self._model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        # Deterministic fallback vectorizer (MD5 hash-based TF-IDF proxy normalized)
        return [self._fallback_embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def _fallback_embed(self, text: str, dim: int = 384) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * dim
        if not tokens:
            return vec
        for token in tokens:
            # Use deterministic MD5 hash instead of python built-in hash() which varies per process
            token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = token_hash % dim
            vec[idx] += 1.0

        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
