"""MLflow query tracing.

Logging is best-effort: a tracking server that is down, misconfigured or simply
not installed must never turn a successful answer into a failed request.
"""

import logging
from typing import Any, Dict, List, Optional

from config import Settings, get_settings

logger = logging.getLogger(__name__)

_configured = False


def _mlflow(settings: Settings):
    """Return a configured mlflow module, or ``None`` when tracing is off."""
    global _configured
    if not settings.mlflow_enabled:
        return None
    try:
        import mlflow
    except ImportError:
        logger.warning("MLFLOW_ENABLED is set but mlflow is not installed; skipping tracing")
        return None

    if not _configured:
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
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
        logger.warning("mlflow logging failed: %s", exc)
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
    """Forget that the tracking URI/experiment were configured (used by tests)."""
    global _configured
    _configured = False
