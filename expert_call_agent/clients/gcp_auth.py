"""Cached, non-blocking Application Default Credentials access.

google.auth.default() and Credentials.refresh() do blocking network/disk I/O,
so they must not run on the event loop. This module resolves ADC once, caches
the credentials, refreshes only when the token has expired, and offloads both
operations to a worker thread.
"""
import asyncio

import google.auth
import google.auth.transport.requests

_credentials = None
_project: str | None = None
_lock = asyncio.Lock()


async def get_auth() -> tuple[str, str]:
    """Return (bearer_token, project), refreshing the token only when expired."""
    global _credentials, _project
    async with _lock:
        if _credentials is None:
            _credentials, _project = await asyncio.to_thread(google.auth.default)
        if not _credentials.valid:
            await asyncio.to_thread(
                _credentials.refresh,
                google.auth.transport.requests.Request(),
            )
        return _credentials.token, (_project or "")
