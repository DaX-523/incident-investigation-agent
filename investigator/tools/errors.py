"""Exceptions raised by tool backends and fault injection; the harness maps them to ToolError."""

from __future__ import annotations

from investigator.tools.models import ErrorKind


class ToolFailure(Exception):
    kind: ErrorKind = "internal"
    retryable: bool = False


class ToolTimeoutError(ToolFailure):
    kind = "timeout"
    retryable = True


class ToolUnavailableError(ToolFailure):
    """Backend error (5xx / 429 style)."""

    kind = "unavailable"
    retryable = True


class ToolNotFoundError(ToolFailure):
    kind = "not_found"
    retryable = False


class ToolInvalidRequestError(ToolFailure):
    kind = "invalid_request"
    retryable = False
