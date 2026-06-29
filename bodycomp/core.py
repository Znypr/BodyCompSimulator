from __future__ import annotations

from dataclasses import dataclass

KCAL_PER_KG_FAT_CHANGE = 9435.0
KCAL_PER_KG_FFM_CHANGE = 1816.0
ADAPTATION_TIME_CONSTANT_DAYS = 21.0
ADAPTATION_FRACTION = 0.12
ALPERT_KCAL_PER_KG_FAT_PER_DAY = 69.3
DAYS_PER_MONTH = 30.4375

ACTIVITY_FACTORS = {
    "Sedentary": 1.20,
    "Light": 1.35,
    "Moderate": 1.55,
    "High": 1.75,
    "Very high": 1.90,
}

MAX_FFM_GAIN_FRACTION_PER_MONTH = {
    "Beginner": 0.0075,
    "Intermediate": 0.0040,
    "Advanced": 0.0020,
}

CYCLE_STRATEGIES = ("Body-fat range", "Fixed duration")


@dataclass(frozen=True)
class ProjectionInputs:
    start_weight_kg: float
    start_body_fat_pct: float
    training_quality: float
    training_status: str
    protein_g_per_kg: float
    bulk_weeks: float
    cut_weeks: float
    surplus_kcal: int
    deficit_kcal: int
    total_weeks: float
    cycle_scale: float
    start_mode: str
    first_phase_weeks: float
    activity_level: str = "Moderate"
    measured_maintenance_kcal: float | None = None
    fixed_intake: bool = True
    cycle_strategy: str = "Fixed duration"
    bulk_stop_body_fat_pct: float = 15.0
    cut_stop_body_fat_pct: float = 10.0
    minimum_phase_weeks: float = 4.0
    finish_lean: bool = False


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def validate_inputs(inputs: ProjectionInputs) -> None:
    if not 30 <= inputs.start_weight_kg <= 300:
        raise ValueError("Start weight must be between 30 and 300 kg.")
    if not 3 <= inputs.start_body_fat_pct <= 65:
        raise ValueError("Body-fat percentage must be between 3 and 65%.")
    if not 0 <= inputs.training_quality <= 1:
        raise ValueError("Training quality must be between 0 and 1.")
    if inputs.training_status not in MAX_FFM_GAIN_FRACTION_PER_MONTH:
        raise ValueError(f"Unknown training status: {inputs.training_status}")
    if inputs.activity_level not in ACTIVITY_FACTORS:
        raise ValueError(f"Unknown activity level: {inputs.activity_level}")
    if inputs.start_mode not in {"Bulk", "Cut"}:
        raise ValueError("Start mode must be 'Bulk' or 'Cut'.")
    if inputs.cycle_strategy not in CYCLE_STRATEGIES:
        raise ValueError(f"Unknown cycle strategy: {inputs.cycle_strategy}")
    if inputs.bulk_weeks <= 0 or inputs.cut_weeks <= 0:
        raise ValueError("Bulk and cut durations must be positive.")
    if inputs.total_weeks <= 0 or inputs.first_phase_weeks <= 0:
        raise ValueError("Timeline and first phase must be positive.")
    if inputs.cycle_scale <= 0:
        raise ValueError("Cycle scale must be positive.")
    if inputs.protein_g_per_kg <= 0:
        raise ValueError("Protein intake must be positive.")
    if inputs.surplus_kcal < 0 or inputs.deficit_kcal < 0:
        raise ValueError("Surplus and deficit must be non-negative.")
    if not 4 <= inputs.cut_stop_body_fat_pct < inputs.bulk_stop_body_fat_pct <= 45:
        raise ValueError("The cut target must be below the bulk ceiling.")
    if inputs.minimum_phase_weeks < 1:
        raise ValueError("Minimum phase duration must be at least one week.")


def cunningham_rmr_kcal(lean_mass_kg: float) -> float:
    if lean_mass_kg <= 0:
        raise ValueError("Lean mass must be positive.")
    return 500.0 + 22.0 * lean_mass_kg


def estimate_maintenance_kcal(
    weight_kg: float,
    body_fat_pct: float,
    activity_level: str = "Moderate",
) -> float:
    if activity_level not in ACTIVITY_FACTORS:
        raise ValueError(f"Unknown activity level: {activity_level}")
    fat_mass = weight_kg * body_fat_pct / 100.0
    lean_mass = weight_kg - fat_mass
    return cunningham_rmr_kcal(lean_mass) * ACTIVITY_FACTORS[activity_level]


def forbes_ffm_weight_fraction(fat_mass_kg: float) -> float:
    safe_fat_mass = max(0.1, fat_mass_kg)
    return 10.4 / (10.4 + safe_fat_mass)


def protein_score(protein_g_per_kg: float, phase: str) -> float:
    target = 2.2 if phase == "Cut" else 1.6
    return clamp(protein_g_per_kg / target, 0.30, 1.05)


def build_fixed_phases(
    total_days: int,
    start_mode: str,
    first_phase_weeks: float,
    bulk_weeks: float,
    cut_weeks: float,
    cycle_scale: float,
) -> list[str]:
    first_days = max(1, round(first_phase_weeks * 7))
    phases = [start_mode] * min(first_days, total_days)
    next_mode = "Cut" if start_mode == "Bulk" else "Bulk"
    cycle_index = 0

    while len(phases) < total_days:
        multiplier = cycle_scale**cycle_index
        duration = bulk_weeks if next_mode == "Bulk" else cut_weeks
        phase_days = max(1, round(duration * 7 * multiplier))
        phases.extend([next_mode] * phase_days)
        next_mode = "Cut" if next_mode == "Bulk" else "Bulk"
        cycle_index += 1

    return phases[:total_days]


def cut_ffm_fraction(
    body_fat_pct: float,
    fat_mass_kg: float,
    weight_kg: float,
    energy_deficit_kcal: float,
    training_quality: float,
    protein_g_per_kg: float,
) -> tuple[float, float, float]:
    weekly_rate = energy_deficit_kcal * 7.0 / (7700.0 * max(weight_kg, 1.0))
    protein = min(protein_score(protein_g_per_kg, "Cut"), 1.0)
    protection = clamp(0.65 * training_quality + 0.35 * protein, 0.0, 1.0)
    leanness_stress = clamp((15.0 - body_fat_pct) / 8.0, 0.0, 1.0)
    rate_stress = clamp((weekly_rate - 0.005) / 0.0075, 0.0, 1.0)

    capacity = max(fat_mass_kg * ALPERT_KCAL_PER_KG_FAT_PER_DAY, 1.0)
    alpert_ratio = energy_deficit_kcal / capacity
    alpert_stress = clamp((alpert_ratio - 0.9) / 0.8, 0.0, 1.0)

    fraction = (
        0.025
        + 0.070 * leanness_stress
        + 0.100 * rate_stress
        + 0.100 * (1.0 - protection)
        + 0.060 * alpert_stress
    )
    return clamp(fraction, 0.02, 0.35), weekly_rate, alpert_ratio


def bulk_daily_ffm_cap(
    weight_kg: float,
    training_status: str,
    training_quality: float,
    protein_g_per_kg: float,
    cumulative_bulk_days: int,
) -> float:
    monthly_rate = MAX_FFM_GAIN_FRACTION_PER_MONTH[training_status]
    protein = min(protein_score(protein_g_per_kg, "Bulk"), 1.0)
    diminishing_returns = 1.0 / (1.0 + cumulative_bulk_days / (365.0 * 4.0))
    monthly_gain = (
        weight_kg
        * monthly_rate
        * training_quality
        * protein
        * diminishing_returns
    )
    return monthly_gain / DAYS_PER_MONTH


def estimated_days_to_cut_target(
    lean_mass_kg: float,
    fat_mass_kg: float,
    target_body_fat_pct: float,
    deficit_kcal: float,
) -> int:
    if deficit_kcal <= 0:
        return 10**9
    target_fraction = target_body_fat_pct / 100.0
    target_fat_mass = target_fraction * lean_mass_kg / (1.0 - target_fraction)
    fat_to_lose = max(0.0, fat_mass_kg - target_fat_mass)
    effective_deficit = max(deficit_kcal * 0.82, 1.0)
    return max(
        7,
        round(fat_to_lose * KCAL_PER_KG_FAT_CHANGE / effective_deficit * 1.08),
    )


def risk_label(phase: str, weekly_rate: float, alpert_ratio: float) -> str:
    if phase != "Cut":
        return "Low"
    if weekly_rate >= 0.0125 or alpert_ratio >= 1.35:
        return "High"
    if weekly_rate >= 0.008 or alpert_ratio >= 1.0:
        return "Moderate"
    return "Low"
