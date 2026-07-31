"""Bearer-token authentication.

Validates JWTs issued by any standards-compliant provider (Cognito, Entra ID /
Azure AD B2C, Auth0, ...). Symmetric ``HS*`` tokens are verified with a shared
secret; asymmetric ``RS*``/``ES*`` tokens are verified against the issuer's JWKS
endpoint. Setting ``AUTH_ENABLED=false`` disables the check for local runs.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from config import ConfigError, Settings, get_settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

ANONYMOUS_SUBJECT = "anonymous"


class Principal(BaseModel):
    """The authenticated caller."""

    subject: str
    scopes: List[str] = []
    claims: Dict[str, Any] = {}

    @property
    def is_anonymous(self) -> bool:
        return self.subject == ANONYMOUS_SUBJECT


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


#: Cap on the JWKS fetch. PyJWT defaults to 30s, which is long enough for an
#: unreachable identity provider to pin a request worker for half a minute.
JWKS_TIMEOUT_SECONDS = 5


@lru_cache(maxsize=4)
def _jwks_client(url: str):
    return jwt.PyJWKClient(url, cache_keys=True, timeout=JWKS_TIMEOUT_SECONDS)


def _signing_key(token: str, settings: Settings) -> str:
    if settings.jwt_is_symmetric:
        return settings.jwt_secret
    try:
        return _jwks_client(settings.jwt_jwks_url).get_signing_key_from_jwt(token).key
    except jwt.PyJWKClientError as exc:
        logger.warning("unable to resolve signing key: %s", exc)
        raise _unauthorized("token signing key could not be resolved") from exc
    except Exception as exc:  # network/TLS failures must fail closed, not 500
        logger.warning("jwks lookup failed: %s", exc)
        raise _unauthorized("token signing key could not be resolved") from exc


def _scopes(claims: Dict[str, Any]) -> List[str]:
    raw = claims.get("scope") or claims.get("scp") or claims.get("permissions") or []
    if isinstance(raw, str):
        return raw.split()
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return []


def decode_token(token: str, settings: Settings) -> Principal:
    """Verify a JWT and return the caller it identifies."""
    options = {
        "verify_aud": bool(settings.jwt_audience),
        "verify_iss": bool(settings.jwt_issuer),
        # PyJWT only checks `exp` when it is present, so a token minted without
        # one would be accepted forever. Bearer tokens have to expire.
        "require": ["exp"],
    }
    try:
        claims = jwt.decode(
            token,
            _signing_key(token, settings),
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience or None,
            issuer=settings.jwt_issuer or None,
            options=options,
        )
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("token has expired") from exc
    except jwt.InvalidTokenError as exc:
        logger.info("rejected token: %s", exc)
        raise _unauthorized("invalid authentication token") from exc

    subject = claims.get("sub") or claims.get("client_id")
    if not subject:
        raise _unauthorized("token is missing a subject claim")
    return Principal(subject=str(subject), scopes=_scopes(claims), claims=claims)


def validate_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings),
) -> Principal:
    """FastAPI dependency that authenticates the request."""
    if not settings.auth_enabled:
        return Principal(subject=ANONYMOUS_SUBJECT)

    try:
        settings.require_auth()
    except ConfigError as exc:
        logger.error("authentication misconfigured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="authentication is not configured",
        ) from exc

    if credentials is None or not credentials.credentials:
        raise _unauthorized("missing bearer token")
    if credentials.scheme.lower() != "bearer":
        raise _unauthorized("unsupported authorization scheme")

    return decode_token(credentials.credentials, settings)
