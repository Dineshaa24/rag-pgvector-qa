"""Text embeddings. Runs locally on CPU with an ONNX model, so no API key is needed."""
from functools import cached_property
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    # BGE models expect this prefix on queries (not on documents) for best retrieval.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def _model(self):
        from fastembed import TextEmbedding
        return TextEmbedding(self.model_name)

    @cached_property
    def dim(self) -> int:
        return len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return list(self._model.embed(texts, batch_size=64))

    def embed_query(self, text: str) -> np.ndarray:
        prefix = self.QUERY_PREFIX if "bge" in self.model_name.lower() else ""
        return next(iter(self._model.embed([prefix + text])))
