"""Chunk storage and search on PostgreSQL: pgvector for semantic search, built-in full-text search
for keywords, and reciprocal rank fusion to combine the two."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


@dataclass
class Chunk:
    doc_id: str
    source: str
    chunk_index: int
    content: str
    content_hash: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    id: int
    doc_id: str
    source: str
    content: str
    score: float


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class PgVectorStore:
    def __init__(self, database_url: str, dim: int):
        self.database_url = database_url
        self.dim = dim

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.database_url, autocommit=True)
        register_vector(conn)
        return conn

    def init_schema(self) -> None:
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        with self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id           bigserial PRIMARY KEY,
                    doc_id       text NOT NULL,
                    source       text NOT NULL,
                    chunk_index  int  NOT NULL,
                    content      text NOT NULL,
                    content_hash text NOT NULL UNIQUE,
                    embedding    vector({self.dim}) NOT NULL,
                    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                         "ON chunks USING hnsw (embedding vector_cosine_ops)")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv)")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id)")

    def add(self, chunks: list[Chunk], embeddings: list[np.ndarray]) -> int:
        """Insert chunks, skipping any whose content is already stored. Returns rows inserted."""
        rows = [(c.doc_id, c.source, c.chunk_index, c.content, c.content_hash, np.asarray(e, dtype=np.float32))
                for c, e in zip(chunks, embeddings)]
        with self._connect() as conn, conn.cursor() as cur:
            before = cur.execute("SELECT count(*) FROM chunks").fetchone()[0]
            cur.executemany("""
                INSERT INTO chunks (doc_id, source, chunk_index, content, content_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING""", rows)
            return cur.execute("SELECT count(*) FROM chunks").fetchone()[0] - before

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def delete_all(self) -> None:
        with self._connect() as conn:
            conn.execute("TRUNCATE chunks RESTART IDENTITY")

    def vector_search(self, query_embedding: np.ndarray, k: int) -> list[Hit]:
        q = np.asarray(query_embedding, dtype=np.float32)
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT id, doc_id, source, content, 1 - (embedding <=> %s) AS score
                FROM chunks ORDER BY embedding <=> %s LIMIT %s""", (q, q, k)).fetchall()
        return [Hit(*r) for r in rows]

    def keyword_search(self, query: str, k: int) -> list[Hit]:
        # OR the terms together so one missing word does not drop a relevant chunk.
        with self._connect() as conn:
            rows = conn.execute("""
                WITH q AS (SELECT replace(plainto_tsquery('english', %s)::text, '&', '|')::tsquery AS query)
                SELECT id, doc_id, source, content, ts_rank_cd(tsv, q.query) AS score
                FROM chunks, q WHERE tsv @@ q.query
                ORDER BY score DESC LIMIT %s""", (query, k)).fetchall()
        return [Hit(*r) for r in rows]

    def hybrid_search(self, query: str, query_embedding: np.ndarray, k: int, candidates: int = 40) -> list[Hit]:
        vec = self.vector_search(query_embedding, candidates)
        kw = self.keyword_search(query, candidates)
        by_id = {h.id: h for h in vec + kw}
        fused = rrf([[h.id for h in vec], [h.id for h in kw]])[:k]
        return [Hit(i, by_id[i].doc_id, by_id[i].source, by_id[i].content, score) for i, score in fused]
