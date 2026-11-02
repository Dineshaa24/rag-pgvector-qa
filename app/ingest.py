"""Split documents into overlapping chunks and store them with their embeddings."""
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.embeddings import Embedder
from app.store import Chunk, PgVectorStore


def split(doc_id: str, source: str, text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = [p.strip() for p in splitter.split_text(text) if p.strip()]
    return [Chunk(doc_id=doc_id, source=source, chunk_index=i, content=p,
                  content_hash=hashlib.sha256(f"{doc_id}\x00{p}".encode()).hexdigest())
            for i, p in enumerate(pieces)]


def ingest(docs: list[dict], store: PgVectorStore, embedder: Embedder, chunk_size: int, chunk_overlap: int) -> dict:
    """docs: [{"doc_id": ..., "source": ..., "text": ...}]"""
    chunks = [c for d in docs for c in split(d["doc_id"], d.get("source", d["doc_id"]), d["text"],
                                               chunk_size, chunk_overlap)]
    if not chunks:
        return {"documents": len(docs), "chunks": 0, "inserted": 0}
    embeddings = embedder.embed_documents([c.content for c in chunks])
    inserted = store.add(chunks, embeddings)
    return {"documents": len(docs), "chunks": len(chunks), "inserted": inserted}
