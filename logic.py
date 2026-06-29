"""Public compatibility API for the body-composition simulator."""

from bodycomp.core import (
    ACTIVITY_FACTORS,
    CYCLE_STRATEGIES,
    MAX_FFM_GAIN_FRACTION_PER_MONTH,
    cunningham_rmr_kcal,
    estimate_maintenance_kcal,
    forbes_ffm_weight_fraction,
)
from bodycomp.report import phase_endpoints, summarize_phases
from bodycomp.simulation import calculate_projection

__all__ = [
    "ACTIVITY_FACTORS",
    "CYCLE_STRATEGIES",
    "MAX_FFM_GAIN_FRACTION_PER_MONTH",
    "calculate_projection",
    "cunningham_rmr_kcal",
    "estimate_maintenance_kcal",
    "forbes_ffm_weight_fraction",
    "phase_endpoints",
    "summarize_phases",
]
