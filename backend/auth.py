from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientConnectionError, PyJWKClientError

_bearer = HTTPBearer(auto_error=False)
_ALLOWED_ASYMMETRIC_ALGORITHMS = {"ES256", "RS256"}


def local_auth_enabled() -> bool:
    auth_required = os.getenv("AUTH_REQUIRED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    host = os.getenv("HOST", "127.0.0.1").strip().lower()
    return not auth_required and host in {"127.0.0.1", "localhost", "::1"}


@lru_cache(maxsize=4)
def _jwks_client(url: str) -> PyJWKClient:
    return PyJWKClient(f"{url}/auth/v1/.well-known/jwks.json", cache_keys=True)


def current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, str]:
    if local_auth_enabled():
        return {"id": "local-user"}
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Sign in to access this resource.")

    supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    if not supabase_url:
        raise HTTPException(503, "Supabase authentication is not configured.")

    token = credentials.credentials
    try:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        if algorithm == "HS256":
            secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
            if not secret:
                raise HTTPException(
                    503,
                    "Configure SUPABASE_JWT_SECRET for this legacy Supabase signing key.",
                )
            signing_key: Any = secret
        elif algorithm in _ALLOWED_ASYMMETRIC_ALGORITHMS:
            signing_key = _jwks_client(supabase_url).get_signing_key_from_jwt(token).key
        else:
            raise jwt.InvalidAlgorithmError("Unsupported signing algorithm.")

        claims = jwt.decode(
            token,
            signing_key,
            algorithms=[algorithm],
            audience="authenticated",
            issuer=f"{supabase_url}/auth/v1",
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except PyJWKClientConnectionError as exc:
        raise HTTPException(503, "Unable to verify sign-in with Supabase right now.") from exc
    except (PyJWKClientError, jwt.InvalidTokenError) as exc:
        raise HTTPException(401, "Your sign-in has expired or is invalid. Please sign in again.") from exc

    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(401, "The sign-in token does not identify a user.")
    return {"id": user_id}
