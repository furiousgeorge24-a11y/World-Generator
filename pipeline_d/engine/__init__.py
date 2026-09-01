"""Public bootstrap state for the pipeline C land-origin module."""

VERSION = "0.1.0-bootstrap"


class EngineUnavailableError(RuntimeError):
    """Raised when generation is requested before an engine exists."""

