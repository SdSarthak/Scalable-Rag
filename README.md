# Scalable RAG

A production-shaped retrieval-augmented generation service: ingest a corpus into
a FAISS index, then serve grounded answers over an authenticated FastAPI
endpoint that scales horizontally on Kubernetes.

- **Ingestion** from a local directory or an S3 prefix, with chunking and
  overlap you control.
- **Retrieval + generation** through LangChain `RetrievalQA` over FAISS, with a
  prompt that forbids answering outside the retrieved context.
- **Auth** via bearer JWTs — HS256 shared secret or RS256/JWKS (Cognito, Entra
  ID, Auth0).
- **Observability** — Prometheus counters, latency histogram, and optional
  MLflow tracing of every query.
- **Deployment** — Dockerfile, Deployment/Service/ConfigMap and an HPA that
  scales 3 → 10 replicas.

## Layout

| Path | What it does |
| --- | --- |
| `config.py` | All settings, read from the environment / `.env` |
| `document_loader.py` | Local and S3 loaders, chunking |
| `vector_store.py` | Build, persist and load the FAISS index |
| `retrieval_chain.py` | `RetrievalService` — the grounded QA pipeline |
| `auth.py` | JWT bearer validation |
| `ml_flow.py` | Best-effort MLflow tracing |
| `main.py` | FastAPI app: `/query`, `/health`, `/ready`, `/metrics` |
| `ingest.py` | CLI that builds the index |
| `k8s/` | Deployment, Service, ConfigMap/Secret template, HPA |
| `tests/` | pytest suite (no network, no API keys required) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in OPENAI_API_KEY at minimum
```

Use a fresh virtualenv: FastAPI and Starlette must come from the same
resolution, and a globally installed mix of the two will fail to import.

## Build the index

```bash
python ingest.py                                   # uses .env
python ingest.py --source local --docs-dir ./docs  # local corpus
python ingest.py --source s3 --bucket my-corpus --prefix docs/
python ingest.py --dry-run                         # count chunks, no embedding
```

The index is written to `INDEX_PATH` (default `faiss_index/`) and is git-ignored.
Re-run the command whenever the corpus changes.

## Run the API

```bash
uvicorn main:app --reload --port 8000
```

`AUTH_ENABLED=false` in `.env` skips token validation while developing.

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "what does the retriever return?", "top_k": 5}'
```

```json
{
  "answer": "The retriever returns the top 5 chunks per question.",
  "sources": [
    {
      "content": "The default retriever returns the top 5 chunks ...",
      "source": "docs/sample-corpus.md",
      "metadata": {"name": "sample-corpus.md", "chunk_id": 3, "start_index": 240}
    }
  ],
  "latency_ms": 812.4,
  "model": "gpt-4o-mini"
}
```

### Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/query` | Authenticated. `question` (required), `top_k` (1-50, optional) |
| `GET` | `/health` | Liveness — the process is up |
| `GET` | `/ready` | Readiness — 503 until the index is loadable |
| `GET` | `/metrics` | Prometheus exposition format |
| `GET` | `/docs` | Interactive OpenAPI docs |

`/query` returns `400` for an unusable question, `401` for a bad or missing
token, `503` when no index has been built, and `502` when the model provider
fails. Upstream error details are logged, never returned to the caller.

## Configuration

Every variable is documented in `.env.example`. The ones you will actually
touch:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required for embedding and generation |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model used to synthesise answers |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for index and query |
| `DOCUMENT_SOURCE` | `local` | `local` or `s3` |
| `DOCS_DIR` / `S3_BUCKET_NAME` | `docs` / — | Where the corpus lives |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Splitter geometry |
| `MAX_DOCUMENT_MB` | `25` | Ingestion skips larger documents; `0` disables the cap |
| `INDEX_PATH` | `faiss_index` | Where the FAISS index is written |
| `RETRIEVAL_K` | `5` | Chunks fed to the model per question |
| `AUTH_ENABLED` | `true` | Set `false` for local development only |
| `JWT_ALGORITHM` | `HS256` | `HS*` uses `JWT_SECRET`, `RS*` uses `JWT_JWKS_URL` |
| `MLFLOW_ENABLED` | `false` | Trace queries to `MLFLOW_TRACKING_URI` |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers config validation, both loaders (including S3 pagination
against a stubbed client), JWT validation, answer serialisation, MLflow
degradation and every API route. It never calls OpenAI, AWS or MLflow.

## Deploy

```bash
docker build -t rag-system:latest .
docker run --rm -p 8000:8000 --env-file .env -v "$PWD/faiss_index:/app/faiss_index" rag-system:latest
```

On Kubernetes, create the secret first, then apply the manifests:

```bash
kubectl create secret generic rag-system-secrets \
  --from-literal=OPENAI_API_KEY=... \
  --from-literal=S3_BUCKET_NAME=...
kubectl apply -f k8s/
```

Pods mount the FAISS index read-only from a `rag-faiss-index` PVC, so replicas
stay stateless and the HPA can scale them freely. Prometheus scrape annotations
are already on the pod template.

## Security notes

- `.env` is git-ignored; only `.env.example` (placeholders) is committed.
- Prefer an IAM role for the service account over static AWS keys on EKS.
- `AUTH_ENABLED=true` fails closed: a misconfigured validator returns `500`
  rather than letting requests through.
