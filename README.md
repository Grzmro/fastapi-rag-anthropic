# RAG Document Assistant — Grounded LLM Q&A with Citations

End-to-end Retrieval-Augmented Generation pipeline: document ingestion and
chunking, embedding-based semantic search over a **Chroma** vector database,
and grounded answer generation with **source citations** via the
**Anthropic API** (LangChain). Served as a **FastAPI** REST API, deployable
to **AWS EC2** with Docker.

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │                 FastAPI (REST)                │
                 └──────────────────────────────────────────────┘
   POST /ingest ──► load (pypdf / text) ──► chunk (RecursiveCharacterTextSplitter)
                                                   │  metadata: source, page, chunk_id
                                                   ▼
                                     Chroma (persistent vector DB)
                                     embeddings: sentence-transformers (local)
                                                   ▲
   POST /ask ────► semantic search (top-k) ────────┘
                        │
                        ▼
        numbered context fragments [1]..[k]
                        │
                        ▼
        Claude (claude-haiku-4-5, structured output)
                        │
                        ▼
        { answer, citations: [{marker, source, page, quote}] }
```

Grounding design:

- The system prompt restricts the model to the retrieved fragments only and
  requires it to say explicitly when the context does not contain the answer.
- The model returns only **fragment markers + verbatim quotes** (structured
  output). Mapping markers back to `source`/`page` happens in code, so the
  model cannot hallucinate filenames or page numbers.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # put your ANTHROPIC_API_KEY here
uvicorn app.main:app --reload --port 8000
```

Open Swagger UI at http://localhost:8000/docs, or use curl:

```bash
# Index a sample document
curl -F "file=@data/samples/solar_system.md" http://localhost:8000/ingest

# Ask a question grounded in the document
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which planet has the largest volcano?", "include_chunks": false}'

# A question the documents cannot answer -> the model should refuse, not hallucinate
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the GDP of France?"}'

# Health & indexed sources
curl http://localhost:8000/health
curl http://localhost:8000/documents
```

## API

| Method | Path         | Description                                        |
|--------|--------------|----------------------------------------------------|
| POST   | `/ingest`    | Upload a PDF/TXT/MD file; chunk + index in Chroma  |
| GET    | `/documents` | List indexed sources and chunk count               |
| POST   | `/ask`       | Grounded answer with citations                     |
| GET    | `/health`    | Status, configured models, index size              |

## Configuration

All settings via environment variables / `.env` (see `.env.example`):

| Variable             | Default                                  | Notes                                   |
|----------------------|------------------------------------------|-----------------------------------------|
| `ANTHROPIC_API_KEY`  | —                                        | required for `/ask`                     |
| `ANTHROPIC_MODEL`    | `claude-haiku-4-5`                       | swap to `claude-sonnet-5` / `claude-opus-4-8` for higher quality |
| `EMBEDDING_MODEL`    | `sentence-transformers/all-MiniLM-L6-v2` | for multilingual docs: `intfloat/multilingual-e5-small` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200`                 | chunking parameters                     |
| `TOP_K`              | `4`                                      | retrieved fragments per question        |
| `CHROMA_PERSIST_DIR` | `./chroma_data`                          | vector index location                   |

## Tests

```bash
pytest -v
```

Automated tests cover the deterministic part of the pipeline (loading,
chunking, metadata, retrieval) and run **without** an Anthropic API key.
Prompt/answer quality is verified manually via Swagger UI (`/docs`); an
automated evaluation loop (groundedness scoring + hallucination flagging with
pandas/NumPy) is planned in `eval/` once the prompt is finalized.

## Docker

```bash
docker compose build
docker compose up -d
```

The Chroma index is persisted in `./chroma_data` on the host. The embedding
model is baked into the image at build time, so startup is fast.

## Deployment on AWS EC2

1. Launch an EC2 instance (`t3.small` recommended — the embedding model needs
   ~1 GB RAM), Amazon Linux 2023 or Ubuntu 22.04.
2. Security group: allow inbound TCP 8000 (or put nginx/Caddy on 80/443).
3. Install Docker + the compose plugin, then:

   ```bash
   git clone https://github.com/<you>/fastapi-rag-anthropic.git
   cd fastapi-rag-anthropic
   cp .env.example .env   # set ANTHROPIC_API_KEY (never commit it)
   docker compose up -d
   ```

4. Verify: `curl http://<ec2-public-ip>:8000/health`.

For production hardening consider AWS SSM Parameter Store / Secrets Manager
for the API key instead of a plaintext `.env` file.

## Tech stack

Python 3.12 · FastAPI · LangChain (`langchain-anthropic`, `langchain-chroma`,
`langchain-huggingface`) · Chroma · sentence-transformers · Anthropic API
(`claude-haiku-4-5`) · pandas/NumPy (eval) · Docker · AWS EC2
