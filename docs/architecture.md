# Architecture

The service has two paths.

**Ingestion (offline).** `ingest.py` reads the corpus from the local filesystem
or from an S3 prefix, splits every document into overlapping character chunks
with `RecursiveCharacterTextSplitter`, embeds the chunks with the configured
OpenAI embedding model and writes a FAISS index to `INDEX_PATH`.

**Serving (online).** The FastAPI app loads that index lazily on the first
request (or during startup warm-up), wraps it in a `RetrievalQA` chain with a
grounded prompt, and answers questions at `POST /query`. Every request is
authenticated with a bearer JWT, counted in Prometheus and optionally traced to
MLflow.

Because the index is read-only at serving time, replicas are stateless: the
Horizontal Pod Autoscaler can scale them from 3 to 10 pods on CPU and memory
utilisation without any coordination between them.
