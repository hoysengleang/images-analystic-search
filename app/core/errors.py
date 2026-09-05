from typing import Any, Optional

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class OpenVisionSearchError(Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "INTERNAL_SERVER_ERROR"
    message = "Internal server error"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.details = details or {}
        super().__init__(self.message)


class BadRequestError(OpenVisionSearchError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "BAD_REQUEST"
    message = "Bad request"


class InvalidImageError(BadRequestError):
    code = "INVALID_IMAGE"
    message = "Invalid image file"


class ImageTooLargeError(BadRequestError):
    code = "IMAGE_TOO_LARGE"
    message = "Image file is too large"


class UnsupportedImageTypeError(BadRequestError):
    code = "UNSUPPORTED_IMAGE_TYPE"
    message = "Unsupported image type"


class UnauthorizedError(OpenVisionSearchError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "UNAUTHORIZED"
    message = "Missing or invalid API key"


class RequestTooLargeError(OpenVisionSearchError):
    status_code = 413
    code = "REQUEST_BODY_TOO_LARGE"
    message = "Request body is too large"


class ResourceNotFoundError(OpenVisionSearchError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "NOT_FOUND"
    message = "Resource not found"


class ConflictError(OpenVisionSearchError):
    status_code = status.HTTP_409_CONFLICT
    code = "CONFLICT"
    message = "Resource conflict"


class ServiceUnavailableError(OpenVisionSearchError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "SERVICE_UNAVAILABLE"
    message = "Service unavailable"


def error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": jsonable_encoder(details or {}),
            }
        },
    )


async def openvisionsearch_error_handler(
    request: Request,
    exc: OpenVisionSearchError,
) -> JSONResponse:
    return error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    status_code = exc.status_code
    code = "HTTP_ERROR"
    message = "HTTP error"

    if status_code == status.HTTP_400_BAD_REQUEST:
        code = "BAD_REQUEST"
        message = "Bad request"
    elif status_code == status.HTTP_401_UNAUTHORIZED:
        code = "UNAUTHORIZED"
        message = "Unauthorized"
    elif status_code == status.HTTP_403_FORBIDDEN:
        code = "FORBIDDEN"
        message = "Forbidden"
    elif status_code == status.HTTP_404_NOT_FOUND:
        code = "NOT_FOUND"
        message = "Resource not found"
    elif status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        code = "METHOD_NOT_ALLOWED"
        message = "Method not allowed"

    if isinstance(exc.detail, str) and exc.detail and status_code < 500:
        message = exc.detail

    return error_response(
        status_code=status_code,
        code=code,
        message=message,
        details={},
    )


async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="VALIDATION_ERROR",
        message="Invalid request",
        details={"errors": exc.errors()},
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_SERVER_ERROR",
        message="Internal server error",
        details={},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(OpenVisionSearchError, openvisionsearch_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
