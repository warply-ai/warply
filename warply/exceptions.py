"""Warply SDK exceptions."""


class WarplyError(Exception):
    """Base exception for Warply SDK errors."""


class ValidationError(WarplyError, ValueError):
    """Raised when a spec object fails validation."""


class NotReadyError(WarplyError, RuntimeError):
    """Raised when an operation requires a running deployment."""
