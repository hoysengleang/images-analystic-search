"""ASGI middleware that runs before request bodies are parsed."""

from __future__ import annotations

from starlette.datastructures import Headers

from app.core.errors import RequestTooLargeError, error_response


class RequestBodySizeLimitMiddleware:
    """Reject oversized requests before the body reaches the framework.

    Route handlers only see a request after the whole body has been received
    and, for uploads, spooled to disk. Enforcing the cap here is what keeps a
    huge upload from consuming memory and disk before any check runs.
    """

    def __init__(self, app, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared_size = self._declared_size(scope)
        if declared_size is not None and declared_size > self.max_body_bytes:
            await self._reject(scope, receive, send, declared_size)
            return

        await self.app(scope, self._counting_receive(receive), send)

    def _declared_size(self, scope):
        content_length = Headers(scope=scope).get("content-length")
        if not content_length:
            return None

        try:
            return int(content_length)
        except ValueError:
            return None

    def _counting_receive(self, receive):
        """Guard chunked uploads, which arrive without a Content-Length."""
        received_bytes = 0

        async def counting_receive():
            nonlocal received_bytes
            message = await receive()

            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise RequestTooLargeError(
                        details={"max_body_bytes": self.max_body_bytes},
                    )

            return message

        return counting_receive

    async def _reject(self, scope, receive, send, declared_size: int) -> None:
        response = error_response(
            status_code=RequestTooLargeError.status_code,
            code=RequestTooLargeError.code,
            message=RequestTooLargeError.message,
            details={
                "content_length": declared_size,
                "max_body_bytes": self.max_body_bytes,
            },
        )
        await response(scope, receive, send)
