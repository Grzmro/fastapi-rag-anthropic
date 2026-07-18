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

| Method | Path            | Description                                        |
|--------|-----------------|----------------------------------------------------|
| POST   | `/ingest`       | Upload a PDF/TXT/MD file; chunk + index in Chroma  |
| POST   | `/ingest/drive` | Index all supported files from a Google Drive folder |
| GET    | `/documents`    | List indexed sources and chunk count               |
| POST   | `/ask`          | Grounded answer with citations                     |
| GET    | `/health`       | Status, configured models, index size              |

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

## Google Drive ingestion (optional)

Index documents straight from a Drive folder instead of uploading them by
hand. Auth uses a **service account** — the app can only read folders you
explicitly share with it.

One-time setup:

1. In [Google Cloud Console](https://console.cloud.google.com) create a
   project and **enable the Google Drive API**.
2. Create a **service account** (IAM & Admin → Service Accounts), then add a
   key (Keys → Add key → JSON) and download the file.
3. Save it in the project root as `gcp-service-account.json` (gitignored) and
   set `GOOGLE_SERVICE_ACCOUNT_FILE=./gcp-service-account.json` in `.env`.
4. In Google Drive, **share the folder** with the service account's e-mail
   address (looks like `name@project.iam.gserviceaccount.com`), Viewer role.

Usage — the folder ID is the part after `/folders/` in the Drive URL:

```bash
curl -X POST http://localhost:8000/ingest/drive \
  -H "Content-Type: application/json" \
  -d '{"folder_id": "1AbCdEfGh..."}'
```

Supported: PDF, TXT, MD and native **Google Docs** (exported as plain text).
Other types (images, spreadsheets, ...) are reported back in `skipped`.

## Tests

```bash
pytest -v
```

Automated tests cover the deterministic part of the pipeline (loading,
chunking, metadata, retrieval) and run **without** an Anthropic API key.

## Evaluation

Automated answer-quality evaluation over the gold set in `eval/gold.jsonl`
(derived from `data/TEST_QUESTIONS.md`: 8 answerable questions + 4 traps):

```bash
python -m eval.run_eval                        # retrieval metrics only — no API key needed
python -m eval.run_eval --stage all --judge    # full run (needs ANTHROPIC_API_KEY)
```

What gets measured:

- **Retrieval** — hit rate @k and MRR of the vector search against the
  expected source document. The hit-rate-vs-k curve also tells you whether a
  reranker would help (high @20 but low @4 ⇒ yes).
- **Generation (deterministic)** — expected facts present in the answer,
  **citation validity** (every quote must literally occur in the cited chunk —
  a hard hallucination check), correct source cited, and **refusal
  correctness** on trap questions (empty citations list expected).
- **Faithfulness (LLM judge, `--judge`)** — a judge model splits the answer
  into claims and marks each as supported by the context or not. The default
  judge is `claude-haiku-4-5` (same as the generator — cheap, but effectively
  self-evaluation); for a final validation use a stronger judge, e.g.
  `--judge-model claude-sonnet-5`.

Per-question results land in `eval/results/<timestamp>_*.csv` (gitignored).
The eval builds its index from `data/samples/` in a temporary directory and
never touches the production `./chroma_data`.

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
