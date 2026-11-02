"""FastAPI service: ingest documents, then ask questions answered from them with citations."""
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import settings
from app.embeddings import FastEmbedEmbedder
from app.ingest import ingest
from app.store import PgVectorStore

state: dict = {}


def build_components():
    """Create the embedder, store and (optional) LLM. Tests replace this with fakes."""
    embedder = FastEmbedEmbedder(settings.embedding_model)
    store = PgVectorStore(settings.database_url, embedder.dim)
    store.init_schema()
    answerer = None
    if settings.llm_enabled:
        from app.llm import Answerer
        answerer = Answerer(settings.llm_model, settings.llm_base_url, settings.llm_api_key)
    return embedder, store, answerer


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["embedder"], state["store"], state["answerer"] = build_components()
    yield
    state.clear()


app = FastAPI(title="RAG Q&A on pgvector", version="1.0.0", lifespan=lifespan)


class Document(BaseModel):
    doc_id: str = Field(..., min_length=1)
    source: str | None = None
    text: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    documents: list[Document] = Field(..., min_length=1, max_length=500)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    k: int = Field(4, ge=1, le=20)
    # Vector search scored best in the SQuAD evaluation (see README), so it is the default.
    mode: Literal["vector", "hybrid", "keyword"] = "vector"


class Source(BaseModel):
    doc_id: str
    source: str
    score: float
    content: str


class AskResponse(BaseModel):
    answer: str | None
    sources: list[Source]
    mode: str
    retrieval_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "chunks": state["store"].count(), "llm": state["answerer"] is not None}


@app.post("/ingest")
def ingest_docs(req: IngestRequest):
    docs = [d.model_dump() for d in req.documents]
    return ingest(docs, state["store"], state["embedder"], settings.chunk_size, settings.chunk_overlap)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    store, embedder, answerer = state["store"], state["embedder"], state["answerer"]
    start = time.perf_counter()
    if req.mode == "keyword":
        hits = store.keyword_search(req.question, req.k)
    elif req.mode == "vector":
        hits = store.vector_search(embedder.embed_query(req.question), req.k)
    else:
        hits = store.hybrid_search(req.question, embedder.embed_query(req.question), req.k)
    retrieval_ms = (time.perf_counter() - start) * 1000

    # Without an LLM configured the service still works as a semantic search API.
    answer = answerer.answer(req.question, hits) if answerer and hits else None
    return AskResponse(
        answer=answer,
        sources=[Source(doc_id=h.doc_id, source=h.source, score=round(h.score, 4), content=h.content) for h in hits],
        mode=req.mode,
        retrieval_ms=round(retrieval_ms, 1),
    )
