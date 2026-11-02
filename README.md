# rag-pgvector-qa

Question answering over your own documents, with PostgreSQL + pgvector as the vector database. You ingest text through an API, ask a question, and get an answer with numbered citations back to the passages it came from.

I wanted a RAG setup that runs on plain PostgreSQL rather than a separate vector database, since that is what I use in production (Amazon Aurora PostgreSQL supports pgvector). Everything here runs locally, and the embeddings need no API key.

## How it works

1. **Ingest:** documents are split into overlapping chunks (LangChain `RecursiveCharacterTextSplitter`, 800 characters, 100 overlap), embedded with `BAAI/bge-small-en-v1.5` (ONNX on CPU via fastembed), and stored in a `chunks` table. A content hash makes ingestion idempotent, so re-sending the same document adds nothing.
2. **Retrieve:** three modes, all inside Postgres:
   - `vector`: cosine similarity on an HNSW index (pgvector)
   - `keyword`: Postgres full-text search (`tsvector` + GIN index)
   - `hybrid`: both, merged with reciprocal rank fusion
3. **Answer:** the top chunks go to a chat model with a prompt that only allows answers from the context and asks for `[1]`-style citations. Any OpenAI-compatible endpoint works (OpenAI, Azure OpenAI, Ollama, vLLM). With no model configured, the API still returns the ranked sources, so it doubles as a semantic search service.

## Retrieval results

I evaluated retrieval on the [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dev set: all 2,067 paragraphs (48 Wikipedia articles, 2,904 chunks) were ingested, and for 2,000 random questions I checked whether the paragraph containing the answer came back.

| Mode | Recall@1 | Recall@5 | MRR@10 | Median latency |
|---|---|---|---|---|
| Keyword (full-text) | 38.0% | 69.0% | 0.514 | 43 ms |
| **Vector (pgvector)** | **67.0%** | **87.5%** | **0.756** | 61 ms |
| Hybrid (RRF) | 64.7% | 87.7% | 0.745 | 103 ms |

What I took from this:

- Dense retrieval is far better than keyword search here, because SQuAD questions are paraphrases of the passage rather than copies of its words.
- Hybrid search did **not** help on this dataset. Fusing in the weaker keyword ranking slightly hurt the top-1 result. It tends to pay off when queries contain exact identifiers (product codes, error messages, names), so it stays available as a mode, but `vector` is the default.
- Latency is measured end to end from Python on Windows to Postgres in Docker, including query embedding and a new connection per query. A connection pool would cut most of it.

Reproduce with `python -m scripts.evaluate --reset`. It writes `results.json`.

## Run it

```bash
docker compose up -d db                      # PostgreSQL 16 with pgvector on port 5433
pip install -r requirements.txt
uvicorn app.main:app --reload
```

To get generated answers, set a model before starting:

```bash
export OPENAI_API_KEY=sk-...                  # OpenAI
# or a local model through Ollama:
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1
```

Or run everything in Docker with `docker compose up --build`. API docs are at `http://localhost:8000/docs`.

## API

`POST /ingest`

```json
{"documents": [{"doc_id": "warsaw", "source": "Warsaw (Wikipedia)", "text": "Warsaw is the capital and largest city of Poland..."}]}
```

returns `{"documents": 1, "chunks": 1, "inserted": 1}`.

`POST /ask`

```json
{"question": "What is the capital of Poland?", "k": 4, "mode": "vector"}
```

returns the answer (or `null` without a model), the ranked sources with scores, and the retrieval time.

`GET /health` shows the chunk count and whether a model is configured.

## Tests

```bash
docker compose up -d db
pytest
```

Unit tests (chunking, rank fusion, prompt building) need nothing. Integration tests ingest documents and query all three modes against a real pgvector database using a small fake embedder, and they skip themselves if no database is reachable.

## Project layout

```
app/
  config.py      settings from environment variables
  embeddings.py  local ONNX embeddings (fastembed)
  store.py       pgvector + full-text search, reciprocal rank fusion
  ingest.py      chunking and idempotent inserts
  llm.py         prompt and OpenAI-compatible chat model via LangChain
  main.py        FastAPI app
scripts/
  evaluate.py    SQuAD retrieval benchmark
tests/
```

## What I'd do next

- Add a cross-encoder reranker on the top 20 results, which usually lifts Recall@1 more than hybrid search does.
- Use a connection pool (`psycopg_pool`) and batch query embeddings.
- Evaluate answer quality (faithfulness, citation accuracy) and not just retrieval, with a small LLM-judged test set.

## License

MIT
