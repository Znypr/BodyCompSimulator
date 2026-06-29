from __future__ import annotations

from dataclasses import dataclass

KCAL_PER_KG_FAT_CHANGE = 9435.0
KCAL_PER_KG_FFM_CHANGE = 1816.0
DAYS_PER_MONTH = 30.4375
ADAPTATION_TIME_CONSTANT_DAYS = 28.0
CUT_ADAPTATION_FRACTION = 0.10
BULK_ADAPTATION_FRACTION = 0.22
TRANSIENT_TIME_CONSTANT_DAYS = 9.0

ACTIVITY_FACTORS = {
    "Sedentary": 1.20,
    "Light": 1.35,
    "Moderate": 1.55,
    "High": 1.75,
    "Very high": 1.90,
}

# Scenario priors, not biological laws. These represent stable fat-free tissue,
# not short-term glycogen/water changes.
MAX_FFM_GAIN_KG_PER_MONTH = {
    "Beginner": (0.35, 0.75),
    "Intermediate": (0.18, 0.45),
    "Advanced": (0.08, 0.25),
}
MAX_FFM_GAIN_FRACTION_PER_MONTH = {
    key: high / 75.0 for key, (_, high) in MAX_FFM_GAIN_KG_PER_MONTH.items()
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
    cut_gap_multiplier: float = 1.0
    include_scale_transients: bool = True


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def validate_inputs(inputs: ProjectionInputs) -> None:
    if not 30 <= inputs.start_weight_kg <= 300:
        raise ValueError("Start weight must be between 30 and 300 kg.")
    if not 3 <= inputs.start_body_fat_pct <= 65:
        raise ValueError("Body-fat percentage must be between 3 and 65%.")
    if not 0 <= inputs.training_quality <= 1:
        raise ValueError("Training quality must be between 0 and 1.")
    if inputs.training_status not in MAX_FFM_GAIN_KG_PER_MONTH:
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
    if not 0.5 <= inputs.cut_gap_multiplier <= 2.5:
        raise ValueError("Cut calibration multiplier must be between 0.5 and 2.5.")
    if inputs.cycle_strategy == "Body-fat range":
        if not 4 <= inputs.cut_stop_body_fat_pct < inputs.bulk_stop_body_fat_pct <= 45:
            raise ValueError("The cut target must be below the bulk ceiling.")


def cunningham_rmr_kcal(lean_mass_kg: float) -> float:
    if lean_mass_kg <= 0:
        raise ValueError("Lean mass must be positive.")
    return 500.0 + 22.0 * lean_mass_kg


def estimate_maintenance_kcal(weight_kg: float, body_fat_pct: float, activity_level: str = "Moderate") -> float:
    if activity_level not in ACTIVITY_FACTORS:
        raise ValueError(f"Unknown activity level: {activity_level}")
    fat_mass = weight_kg * body_fat_pct / 100.0
    lean_mass = weight_kg - fat_mass
    return cunningham_rmr_kcal(lean_mass) * ACTIVITY_FACTORS[activity_level]


def forbes_ffm_weight_fraction(fat_mass_kg: float) -> float:
    """Historical reference only; not used as the tissue-partition engine."""
    return 10.4 / (10.4 + max(0.1, fat_mass_kg))


def protein_score(protein_g_per_kg: float, phase: str) -> float:
    target = 2.0 if phase == "Cut" else 1.6
    return clamp(protein_g_per_kg / target, 0.3, 1.05)


def cut_ffm_fraction(
    body_fat_pct: float,
    weight_kg: float,
    effective_deficit_kcal: float,
    training_quality: float,
    protein_g_per_kg: float,
) -> tuple[float, float]:
    """Fraction of stable tissue loss assigned to fat-free tissue.

    For lean resistance-trained users with adequate protein and moderate loss
    rates, the central estimate is deliberately low. It increases continuously
    with leanness, aggressive loss rate, weak training and insufficient protein.
    """
    weekly_rate = effective_deficit_kcal * 7.0 / (8500.0 * max(weight_kg, 1.0))
    protection = 0.65 * training_quality + 0.35 * min(protein_score(protein_g_per_kg, "Cut"), 1.0)
    leanness = clamp((14.0 - body_fat_pct) / 7.0, 0.0, 1.0)
    aggression = clamp((weekly_rate - 0.006) / 0.008, 0.0, 1.0)
    fraction = 0.018 + 0.055 * leanness + 0.09 * aggression + 0.10 * (1.0 - protection)
    return clamp(fraction, 0.015, 0.30), weekly_rate


def bulk_ffm_gain_cap_kg_per_day(inputs: ProjectionInputs, stable_weight_kg: float, cumulative_bulk_days: int) -> float:
    low, high = MAX_FFM_GAIN_KG_PER_MONTH[inputs.training_status]
    quality = clamp(inputs.training_quality, 0.0, 1.0)
    protein = min(protein_score(inputs.protein_g_per_kg, "Bulk"), 1.0)
    monthly = low + (high - low) * quality
    monthly *= protein
    monthly *= clamp(stable_weight_kg / 75.0, 0.75, 1.30)
    monthly *= 1.0 / (1.0 + cumulative_bulk_days / 1460.0)
    return monthly / DAYS_PER_MONTH


def estimate_cut_transient_target_kg(weight_kg: float, deficit_kcal: float) -> float:
    """Central short-term scale reduction from glycogen, associated water and gut content."""
    target = 0.015 * weight_kg + 0.0015 * deficit_kcal
    return clamp(target, 0.8, 3.0)


def estimate_bulk_transient_target_kg(weight_kg: float, surplus_kcal: float) -> float:
    target = 0.007 * weight_kg + 0.0010 * surplus_kcal
    return clamp(target, 0.4, 1.5)


def infer_cut_calibration(
    observed_start_weight_kg: float,
    observed_end_weight_kg: float,
    observed_weeks: float,
    planned_deficit_kcal: float,
) -> dict[str, float]:
    """Infer the effective average deficit from an observed scale trend.

    The estimate removes a central transient scale component before converting
    remaining loss to energy using 8,500 kcal/kg, appropriate for predominantly
    fat tissue loss with a small FFM component. It is a calibration estimate,
    not proof of exact intake or expenditure.
    """
    if observed_weeks <= 0 or planned_deficit_kcal <= 0:
        raise ValueError("Observed weeks and planned deficit must be positive.")
    scale_loss = observed_start_weight_kg - observed_end_weight_kg
    transient = min(max(scale_loss, 0.0), estimate_cut_transient_target_kg(observed_start_weight_kg, planned_deficit_kcal))
    tissue_loss = max(0.0, scale_loss - transient)
    effective_deficit = tissue_loss * 8500.0 / (observed_weeks * 7.0)
    multiplier = clamp(effective_deficit / planned_deficit_kcal, 0.5, 2.5)
    return {
        "scale_loss_kg": scale_loss,
        "transient_loss_kg": transient,
        "estimated_tissue_loss_kg": tissue_loss,
        "effective_deficit_kcal": effective_deficit,
        "cut_gap_multiplier": multiplier,
    }


def build_fixed_phases(total_days: int, start_mode: str, first_phase_weeks: float, bulk_weeks: float, cut_weeks: float, cycle_scale: float) -> list[str]:
    phases = [start_mode] * min(max(1, round(first_phase_weeks * 7)), total_days)
    next_mode = "Cut" if start_mode == "Bulk" else "Bulk"
    cycle_index = 0
    while len(phases) < total_days:
        duration = bulk_weeks if next_mode == "Bulk" else cut_weeks
        phase_days = max(1, round(duration * 7 * cycle_scale**cycle_index))
        phases.extend([next_mode] * phase_days)
        next_mode = "Cut" if next_mode == "Bulk" else "Bulk"
        cycle_index += 1
    return phases[:total_days]
