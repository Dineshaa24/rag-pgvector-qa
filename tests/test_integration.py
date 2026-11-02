"""End-to-end tests against a real PostgreSQL + pgvector database.

They use a fake bag-of-words embedder so no model download is needed, and are skipped
when TEST_DATABASE_URL is not reachable. Start one with: docker compose up -d db
"""
import hashlib
import os

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

TEST_DB = os.getenv("TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/rag_test")


def _db_available() -> bool:
    admin = TEST_DB.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=2) as conn:
            name = TEST_DB.rsplit("/", 1)[1]
            if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
                conn.execute(f'CREATE DATABASE "{name}"')
        return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="test database not reachable")


class FakeEmbedder:
    """Hashes words into a fixed-size vector, so texts sharing words end up close together."""
    dim = 64

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().replace("?", " ").replace(".", " ").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeAnswerer:
    def answer(self, question, hits):
        return f"answer from [1] {hits[0].source}"


DOCS = [
    {"doc_id": "warsaw", "source": "Warsaw", "text": "Warsaw is the capital and largest city of Poland."},
    {"doc_id": "tesla", "source": "Nikola Tesla", "text": "Nikola Tesla was an inventor born in 1856 in Smiljan."},
    {"doc_id": "steam", "source": "Steam engine", "text": "A steam engine uses steam pressure to do mechanical work."},
]


@pytest.fixture
def client(monkeypatch):
    from app import main
    from app.store import PgVectorStore

    def fake_components():
        emb = FakeEmbedder()
        store = PgVectorStore(TEST_DB, emb.dim)
        with psycopg.connect(TEST_DB, autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS chunks")
        store.init_schema()
        return emb, store, FakeAnswerer()

    monkeypatch.setattr(main, "build_components", fake_components)
    with TestClient(main.app) as c:
        yield c


def test_ingest_is_idempotent(client):
    first = client.post("/ingest", json={"documents": DOCS}).json()
    again = client.post("/ingest", json={"documents": DOCS}).json()
    assert first["inserted"] == 3 and again["inserted"] == 0
    assert client.get("/health").json()["chunks"] == 3


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_ask_finds_the_right_document(client, mode):
    client.post("/ingest", json={"documents": DOCS})
    body = client.post("/ask", json={"question": "Which city is the capital of Poland?", "mode": mode, "k": 2}).json()
    assert body["sources"][0]["doc_id"] == "warsaw"
    assert body["answer"] == "answer from [1] Warsaw"


def test_validation(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "magic"}).status_code == 422
    assert client.post("/ingest", json={"documents": []}).status_code == 422
