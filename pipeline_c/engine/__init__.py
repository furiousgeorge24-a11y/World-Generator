"""Public state for the Pipeline C land-origin laboratory."""

VERSION = "0.3.0-c4-foundation"


class EngineUnavailableError(RuntimeError):
    """Raised when generation is requested before an engine exists."""
