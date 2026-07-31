"""MLflow query tracing.

Logging is best-effort: a tracking server that is down, misconfigured or simply
not installed must never turn a successful answer into a failed request.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from config import Settings, get_settings

logger = logging.getLogger(__name__)

#: How long tracing stays switched off after a failure. Without this a dead
#: tracking server is re-contacted on every single query and its connection
#: timeout is added to each answer's latency.
FAILURE_BACKOFF_SECONDS = 60.0

_lock = threading.Lock()
_configured = False
_suspended_until = 0.0


def _suspend(reason: str, exc: Exception) -> None:
    """Switch tracing off for a while after a failure."""
    global _suspended_until
    _suspended_until = time.monotonic() + FAILURE_BACKOFF_SECONDS
    logger.warning(
        "mlflow %s (%s); tracing suspended for %.0fs",
        reason,
        exc,
        FAILURE_BACKOFF_SECONDS,
    )


def _mlflow(settings: Settings):
    """Return a configured mlflow module, or ``None`` when tracing is off.

    Never raises: configuring MLflow can talk to the tracking server, and a
    server that is down must not turn a good answer into a failed request.
    """
    global _configured
    if not settings.mlflow_enabled:
        return None
    if time.monotonic() < _suspended_until:
        return None
    try:
        import mlflow
    except ImportError as exc:
        _suspend("is not installed", exc)
        return None

    with _lock:
        if not _configured:
            try:
                if settings.mlflow_tracking_uri:
                    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                mlflow.set_experiment(settings.mlflow_experiment)
            except Exception as exc:  # unreachable/misconfigured tracking server
                _suspend("could not be configured", exc)
                return None
            _configured = True
    return mlflow


def log_query(
    question: str,
    answer: str,
    sources: List[Any],
    latency_ms: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> bool:
    """Record one query/answer pair. Returns True when the run was written."""
    settings = settings or get_settings()
    mlflow = _mlflow(settings)
    if mlflow is None:
        return False

    try:
        with mlflow.start_run(run_name="query"):
            mlflow.log_param("model", settings.llm_model)
            mlflow.log_param("retrieval_k", settings.retrieval_k)
            mlflow.log_metric("source_count", len(sources))
            mlflow.log_metric("question_length", len(question))
            mlflow.log_metric("answer_length", len(answer))
            if latency_ms is not None:
                mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_text(question, "question.txt")
            mlflow.log_text(answer, "answer.txt")
            mlflow.log_text("\n".join(_source_names(sources)), "sources.txt")
        return True
    except Exception as exc:  # tracing must never break the request path
        _suspend("logging failed", exc)
        return False


def _source_names(sources: List[Any]) -> List[str]:
    names = []
    for source in sources:
        if isinstance(source, dict):
            names.append(str(source.get("source", "unknown")))
        else:
            metadata: Dict[str, Any] = getattr(source, "metadata", {}) or {}
            names.append(str(metadata.get("source", "unknown")))
    return names


def reset_tracing_state() -> None:
    """Forget the configuration and any failure backoff (used by tests)."""
    global _configured, _suspended_until
    _configured = False
    _suspended_until = 0.0
