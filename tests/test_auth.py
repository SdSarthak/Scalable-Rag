import datetime as dt

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from auth import ANONYMOUS_SUBJECT, decode_token, validate_token
from conftest import make_settings

SECRET = "unit-test-secret"


def make_token(**claims):
    payload = {
        "sub": "user-123",
        "exp": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5),
    }
    payload.update(claims)
    return jwt.encode(payload, SECRET, algorithm="HS256")


def bearer(token):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_valid_token_yields_a_principal():
    settings = make_settings(auth_enabled=True, jwt_secret=SECRET)
    principal = validate_token(bearer(make_token(scope="read write")), settings)

    assert principal.subject == "user-123"
    assert principal.scopes == ["read", "write"]
    assert not principal.is_anonymous


def test_list_scopes_are_supported():
    settings = make_settings(auth_enabled=True, jwt_secret=SECRET)
    principal = decode_token(make_token(permissions=["query:read"]), settings)
    assert principal.scopes == ["query:read"]


def test_expired_token_is_rejected():
    settings = make_settings(auth_enabled=True, jwt_secret=SECRET)
    expired = make_token(exp=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1))
    with pytest.raises(HTTPException) as excinfo:
        validate_token(bearer(expired), settings)
    assert excinfo.value.status_code == 401


def test_token_signed_with_another_secret_is_rejected():
    settings = make_settings(auth_enabled=True, jwt_secret=SECRET)
    foreign = jwt.encode({"sub": "user-123"}, "other-secret", algorithm="HS256")
    with pytest.raises(HTTPException) as excinfo:
        validate_token(bearer(foreign), settings)
    assert excinfo.value.status_code == 401


def test_token_without_subject_is_rejected():
    settings = make_settings(auth_enabled=True, jwt_secret=SECRET)
    token = jwt.encode({"foo": "bar"}, SECRET, algorithm="HS256")
    with pytest.raises(HTTPException) as excinfo:
        validate_token(bearer(token), settings)
    assert excinfo.value.status_code == 401


def test_audience_is_enforced_when_configured():
    settings = make_settings(auth_enabled=True, jwt_secret=SECRET, jwt_audience="rag-api")
    with pytest.raises(HTTPException):
        validate_token(bearer(make_token(aud="other-api")), settings)
    principal = validate_token(bearer(make_token(aud="rag-api")), settings)
    assert principal.subject == "user-123"


def test_missing_credentials_are_rejected():
    settings = make_settings(auth_enabled=True, jwt_secret=SECRET)
    with pytest.raises(HTTPException) as excinfo:
        validate_token(None, settings)
    assert excinfo.value.status_code == 401
    assert excinfo.value.headers["WWW-Authenticate"] == "Bearer"


def test_misconfigured_auth_fails_closed_with_500():
    settings = make_settings(auth_enabled=True, jwt_secret="")
    with pytest.raises(HTTPException) as excinfo:
        validate_token(bearer(make_token()), settings)
    assert excinfo.value.status_code == 500


def test_disabled_auth_returns_anonymous():
    settings = make_settings(auth_enabled=False)
    assert validate_token(None, settings).subject == ANONYMOUS_SUBJECT
