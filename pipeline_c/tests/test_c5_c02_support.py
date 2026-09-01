"""Process-local expensive-fixture cache shared by C02 test modules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys


PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))


@lru_cache(maxsize=1)
def development_cohort():
    """Build the frozen twelve-member C02 cohort at most once per test process."""

    from engine.tectonic_fabric_c02.cohort import build_development_fabric_cohort

    return build_development_fabric_cohort()


@lru_cache(maxsize=1)
def cohort_audits():
    """Audit the cached cohort at most once when the production API is available."""

    from engine.tectonic_fabric_c02.audit import run_cohort_audits

    # One representative world has a full reversed replay in the focused
    # engine tests.  Keep the complete-cohort audit pass practical here; the
    # canonical evidence builder owns the full replay bundle.
    return run_cohort_audits(development_cohort(), verify_replay=False)


@lru_cache(maxsize=1)
def canonical_cohort_audits():
    """Run the authoritative twelve-member reversed-replay gate once."""

    from engine.tectonic_fabric_c02.audit import run_cohort_audits

    return run_cohort_audits(development_cohort(), verify_replay=True)
