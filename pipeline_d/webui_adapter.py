"""Fail-closed shared-WebUI adapter for the Run 1 bootstrap.

There is deliberately no placeholder world, image, or report.  Keeping all
required adapter functions present lets integration be tested without making
the bootstrap look like generator evidence.
"""

from engine import EngineUnavailableError
from engine import registry


def meta() -> dict:
    return registry.meta()


def _unavailable() -> None:
    raise EngineUnavailableError(
        "pipeline_c is a Run 1 bootstrap: no land-origin engine exists yet"
    )


def generate(seed: int, controls: dict, size: int):
    _unavailable()


def views(world):
    _unavailable()


def render_png(world, view: str) -> bytes:
    _unavailable()


def report(world) -> dict:
    _unavailable()

