"""FastAPI entrypoint for the scalable RAG service."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from auth import Principal, validate_token
from config import ConfigError, Settings, get_settings
from ml_flow import log_query
from retrieval_chain import RetrievalService, get_retrieval_service
from vector_store import VectorStoreNotFound

logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("query_requests_total", "Total query requests")
ERROR_COUNT = Counter("query_errors_total", "Total failed query requests", ["reason"])
QUERY_LATENCY = Histogram("query_latency_seconds", "End-to-end query latency in seconds")


#: Hard ceiling on the request body's question field. ``MAX_QUESTION_LENGTH``
#: is the configurable limit; this one exists so an absurd payload is rejected
#: by the parser instead of being loaded, decoded and validated first.
MAX_QUESTION_CHARS = 32_000


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="Natural-language question",
    )
    top_k: Optional[int] = Field(
        None, ge=1, le=50, description="Override how many chunks are retrieved"
    )


class SourceDocument(BaseModel):
    content: str
    source: str
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    latency_ms: float
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("starting %s (source=%s)", settings.app_name, settings.document_source)
    if get_retrieval_service().warmup():
        logger.info("retrieval chain ready")
    else:
        logger.warning("retrieval chain not ready; /query will return 503 until it is")
    yield
    logger.info("shutting down %s", settings.app_name)


app = FastAPI(
    title="Scalable RAG",
    description="Retrieval-augmented question answering over an indexed corpus.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    return response


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    background: BackgroundTasks,
    principal: Principal = Depends(validate_token),
    service: RetrievalService = Depends(get_retrieval_service),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Answer a question from the indexed corpus.

    Declared ``def`` rather than ``async def`` on purpose: ``service.answer``
    makes blocking HTTP calls to the embedding and chat endpoints. On the event
    loop those calls stalled every other request in the process — including the
    liveness probe, which is how a busy pod got itself restarted.
    """
    REQUEST_COUNT.inc()
    started = time.perf_counter()
    try:
        result = service.answer(request.question, top_k=request.top_k)
    except (VectorStoreNotFound, ConfigError) as exc:
        ERROR_COUNT.labels(reason="unavailable").inc()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ValueError as exc:
        ERROR_COUNT.labels(reason="bad_request").inc()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        ERROR_COUNT.labels(reason="upstream").inc()
        logger.exception("query failed for subject=%s", principal.subject)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="the retrieval chain failed to produce an answer",
        ) from exc
    finally:
        QUERY_LATENCY.observe(time.perf_counter() - started)

    # Tracing runs after the response is sent so a slow tracking server never
    # shows up in the caller's latency.
    background.add_task(
        log_query,
        request.question,
        result["answer"],
        result["sources"],
        latency_ms=result.get("latency_ms"),
        settings=settings,
    )
    return QueryResponse(**result)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness probe: the process is up."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready(service: RetrievalService = Depends(get_retrieval_service)) -> JSONResponse:
    """Readiness probe: the index is loadable and the chain can serve traffic."""
    is_ready = service.is_ready()
    return JSONResponse(
        status_code=status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ready" if is_ready else "index_unavailable"},
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
