from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse, Response

TRUSTED_EXECUTION_HEADER = "X-PulseGraph-Trust"
TRUSTED_EXECUTION_VALUE = "trusted-local-code"


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host == "testclient":
        return True
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_local_origin(origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and _is_loopback_host(parsed.hostname)


async def enforce_local_boundary(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    client_host = request.client.host if request.client else None
    if not _is_loopback_host(client_host):
        return JSONResponse(status_code=403, content={"detail": "PulseGraph only accepts local connections."})

    origin = request.headers.get("origin")
    if origin and not _is_local_origin(origin):
        return JSONResponse(status_code=403, content={"detail": "PulseGraph rejected a non-local browser origin."})
    return await call_next(request)


def require_trusted_execution(request: Request) -> None:
    if request.headers.get(TRUSTED_EXECUTION_HEADER) != TRUSTED_EXECUTION_VALUE:
        raise HTTPException(
            status_code=428,
            detail="This endpoint executes local Python. Confirm trusted-local execution before continuing.",
        )
