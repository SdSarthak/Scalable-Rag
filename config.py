"""Central configuration for the RAG service.

Every tunable value lives here and is sourced from environment variables (or a
local ``.env`` file), so nothing operational is hardcoded in the modules.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when the runtime configuration is incomplete or inconsistent."""


#: Algorithms the service will verify tokens with. ``none`` is deliberately
#: absent: with an empty key PyJWT happily accepts unsigned tokens, so a typo
#: in JWT_ALGORITHM must stop the process rather than open the door.
SUPPORTED_JWT_ALGORITHMS = (
    "HS256",
    "HS384",
    "HS512",
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES384",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- application -----------------------------------------------------
    app_name: str = "scalable-rag"
    log_level: str = "INFO"
    cors_allow_origins: str = "*"

    # --- language model --------------------------------------------------
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    embedding_model: str = "text-embedding-3-small"

    # --- documents -------------------------------------------------------
    document_source: str = "local"  # "local" or "s3"
    docs_dir: str = "docs"
    document_extensions: str = ".txt,.md,.markdown,.rst,.pdf"
    s3_bucket_name: str = ""
    s3_prefix: str = "docs/"
    aws_region: str = "us-east-1"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    #: Per-document ceiling; each file is read into memory whole, so without a
    #: cap a single huge object can OOM the ingest job. 0 disables the check.
    max_document_mb: int = 25

    # --- retrieval -------------------------------------------------------
    index_path: str = "faiss_index"
    retrieval_k: int = 5
    max_question_length: int = 2000

    # --- auth ------------------------------------------------------------
    auth_enabled: bool = True
    jwt_algorithm: str = "HS256"
    jwt_secret: str = ""
    jwt_jwks_url: str = ""
    jwt_audience: str = ""
    jwt_issuer: str = ""

    # --- observability ---------------------------------------------------
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "scalable-rag"

    @field_validator("document_source")
    @classmethod
    def _check_source(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"local", "s3"}:
            raise ValueError("document_source must be 'local' or 's3'")
        return value

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unsupported log_level: {value}")
        return value

    @field_validator("chunk_size", "retrieval_k", "max_question_length")
    @classmethod
    def _check_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be greater than zero")
        return value

    @field_validator("jwt_algorithm")
    @classmethod
    def _check_algorithm(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in SUPPORTED_JWT_ALGORITHMS:
            raise ValueError(
                f"unsupported jwt_algorithm {value!r}; expected one of "
                f"{', '.join(SUPPORTED_JWT_ALGORITHMS)}"
            )
        return normalized

    @field_validator("max_document_mb")
    @classmethod
    def _check_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_document_mb cannot be negative")
        return value

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    # --- derived helpers -------------------------------------------------
    @property
    def allowed_origins(self) -> List[str]:
        return [item.strip() for item in self.cors_allow_origins.split(",") if item.strip()]

    @property
    def allowed_extensions(self) -> List[str]:
        return [
            item.strip().lower() if item.strip().startswith(".") else "." + item.strip().lower()
            for item in self.document_extensions.split(",")
            if item.strip()
        ]

    @property
    def max_document_bytes(self) -> int:
        """The per-document size cap in bytes; 0 means unlimited."""
        return self.max_document_mb * 1024 * 1024

    def require_llm(self) -> None:
        """Fail fast when the model provider is not configured."""
        if not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set; the retrieval chain cannot start")

    def require_ingestion(self) -> None:
        if self.document_source == "s3" and not self.s3_bucket_name:
            raise ConfigError("S3_BUCKET_NAME is required when DOCUMENT_SOURCE=s3")

    @property
    def jwt_is_symmetric(self) -> bool:
        return self.jwt_algorithm.startswith("HS")

    def require_auth(self) -> None:
        if not self.auth_enabled:
            return
        if self.jwt_is_symmetric:
            if not self.jwt_secret:
                raise ConfigError(
                    "JWT_SECRET is required for symmetric (HS*) token validation"
                )
        elif not self.jwt_jwks_url:
            # RS*, ES* and PS* all resolve their key from the issuer's JWKS.
            raise ConfigError(
                f"JWT_JWKS_URL is required for asymmetric ({self.jwt_algorithm}) "
                "token validation"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings; used by tests and by the ingestion CLI."""
    get_settings.cache_clear()
