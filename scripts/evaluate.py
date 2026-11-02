"""Load the SQuAD v1.1 dev set into pgvector and measure retrieval quality.

Each SQuAD paragraph becomes a document. For a sample of questions we check whether the
paragraph that contains the answer comes back in the top k, for vector, keyword and hybrid search.

Usage: python -m scripts.evaluate [--questions 2000] [--reset]
"""
import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

from app.config import settings
from app.embeddings import FastEmbedEmbedder
from app.ingest import ingest
from app.store import PgVectorStore

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def load_squad(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL, path)
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    docs, questions = [], []
    for article in data:
        for p_index, para in enumerate(article["paragraphs"]):
            doc_id = f"{article['title']}#{p_index}"
            docs.append({"doc_id": doc_id, "source": article["title"].replace("_", " "), "text": para["context"]})
            questions += [(qa["question"], doc_id) for qa in para["qas"]]
    return docs, questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/squad-dev-v1.1.json")
    ap.add_argument("--questions", type=int, default=2000)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    embedder = FastEmbedEmbedder(settings.embedding_model)
    store = PgVectorStore(settings.database_url, embedder.dim)
    store.init_schema()
    if args.reset:
        store.delete_all()

    docs, questions = load_squad(Path(args.data))
    start = time.time()
    stats = ingest(docs, store, embedder, settings.chunk_size, settings.chunk_overlap)
    print(f"ingested {stats} in {time.time() - start:.0f}s, store has {store.count()} chunks")

    random.seed(42)
    sample = random.sample(questions, min(args.questions, len(questions)))
    results = {}
    for mode in ("keyword", "vector", "hybrid"):
        ranks, latencies = [], []
        for question, gold in sample:
            t = time.perf_counter()
            if mode == "keyword":
                hits = store.keyword_search(question, 10)
            elif mode == "vector":
                hits = store.vector_search(embedder.embed_query(question), 10)
            else:
                hits = store.hybrid_search(question, embedder.embed_query(question), 10)
            latencies.append((time.perf_counter() - t) * 1000)
            ids = [h.doc_id for h in hits]
            ranks.append(ids.index(gold) + 1 if gold in ids else None)
        results[mode] = {
            "recall@1": round(sum(r == 1 for r in ranks) / len(ranks), 4),
            "recall@5": round(sum(r is not None and r <= 5 for r in ranks) / len(ranks), 4),
            "mrr@10": round(sum(1 / r for r in ranks if r) / len(ranks), 4),
            "median_ms": round(statistics.median(latencies), 1),
        }
        print(mode, results[mode])

    out = {"documents": len(docs), "chunks": store.count(), "questions": len(sample),
           "embedding_model": settings.embedding_model, "results": results}
    Path("results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
