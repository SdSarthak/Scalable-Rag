# Sample corpus

This file exists so a fresh checkout has something to index. Point `DOCS_DIR` at
your own directory (or switch `DOCUMENT_SOURCE` to `s3`) once you have a real
corpus.

Facts the demo can answer:

- The retrieval service exposes `POST /query`, `GET /health`, `GET /ready` and
  `GET /metrics`.
- Chunks default to 1000 characters with a 200 character overlap.
- The default retriever returns the top 5 chunks per question.
- Answers are grounded: if the retrieved context does not contain the answer,
  the model is instructed to say that it does not know.
