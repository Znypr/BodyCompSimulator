"""Evidence-informed body-composition projection model.

The model is intentionally transparent rather than pretending to be a clinical
predictor. It combines:

* Cunningham resting metabolic rate from fat-free mass.
* A dynamic expenditure response as body mass and lean mass change.
* Hall-style adaptive thermogenesis with a 14-day time constant.
* Forbes' body-composition relationship as a baseline estimate of the fraction
  of weight change that is fat-free mass.
* Protein, resistance-training quality, deficit severity and training status as
  explicit modifiers.
* The Alpert fat-energy-transfer estimate as a *risk signal*, not a hard cutoff.

All public inputs and outputs use metric units.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

KCAL_PER_KG_FAT_CHANGE = 9435.0
KCAL_PER_KG_FFM_CHANGE = 1816.0
FORBES_CONSTANT_KG = 10.4
ADAPTATION_TIME_CONSTANT_DAYS = 14.0
ADAPTATION_FRACTION = 0.14
ALPERT_KCAL_PER_KG_FAT_PER_DAY = 69.3
DAYS_PER_MONTH = 30.4375

ACTIVITY_FACTORS = {
    "Sedentary": 1.20,
    "Light": 1.35,
    "Moderate": 1.55,
    "High": 1.75,
    "Very high": 1.90,
}

# Transparent priors, not biological ceilings. They cap projected fat-free-mass
# accrual so the model does not turn a large surplus into implausible muscle gain.
MAX_FFM_GAIN_FRACTION_PER_MONTH = {
    "Beginner": 0.008,
    "Intermediate": 0.004,
    "Advanced": 0.002,
}


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


def cunningham_rmr_kcal(lean_mass_kg: float) -> float:
    """Estimate resting metabolic rate in kcal/day from fat-free mass."""
    if lean_mass_kg <= 0:
        raise ValueError("Lean mass must be positive.")
    return 500.0 + 22.0 * lean_mass_kg


def estimate_maintenance_kcal(
    weight_kg: float,
    body_fat_pct: float,
    activity_level: str = "Moderate",
) -> float:
    """Estimate maintenance energy expenditure in kcal/day."""
    if activity_level not in ACTIVITY_FACTORS:
        raise ValueError(f"Unknown activity level: {activity_level}")
    fat_mass = weight_kg * body_fat_pct / 100.0
    lean_mass = weight_kg - fat_mass
    return cunningham_rmr_kcal(lean_mass) * ACTIVITY_FACTORS[activity_level]


def forbes_ffm_weight_fraction(fat_mass_kg: float) -> float:
    """Baseline fraction of a weight change predicted to be fat-free mass.

    Forbes' derivative is a relationship between *changes in body weight*, not a
    direct muscle-gain efficiency score and not a fraction of energy.
    """
    safe_fat_mass = max(0.1, fat_mass_kg)
    return FORBES_CONSTANT_KG / (FORBES_CONSTANT_KG + safe_fat_mass)


def _protein_score(protein_g_per_kg: float, phase: str) -> float:
    target = 2.0 if phase == "Cut" else 1.6
    return clamp(protein_g_per_kg / target, 0.50, 1.05)


def _build_phases(
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
        durations = {
            "Bulk": max(1, round(bulk_weeks * 7 * multiplier)),
            "Cut": max(1, round(cut_weeks * 7 * multiplier)),
        }
        phases.extend([next_mode] * durations[next_mode])
        next_mode = "Cut" if next_mode == "Bulk" else "Bulk"
        cycle_index += 1

    return phases[:total_days]


def _cut_ffm_fraction(
    fat_mass_kg: float,
    weight_kg: float,
    energy_deficit_kcal: float,
    training_quality: float,
    protein_g_per_kg: float,
) -> tuple[float, float, float]:
    """Return FFM fraction, estimated weekly loss rate and Alpert stress ratio."""
    base = forbes_ffm_weight_fraction(fat_mass_kg)
    weekly_rate = energy_deficit_kcal * 7.0 / (7700.0 * max(weight_kg, 1.0))
    protein = _protein_score(protein_g_per_kg, "Cut")
    protection = 0.55 + 0.25 * training_quality + 0.20 * min(protein, 1.0)

    # Little severity penalty below ~0.5% body weight/week; progressively more
    # above that. This is deliberately continuous rather than a hard threshold.
    severity = 1.0 + 0.70 * clamp((weekly_rate - 0.005) / 0.0075, 0.0, 1.0)

    alpert_capacity = max(fat_mass_kg * ALPERT_KCAL_PER_KG_FAT_PER_DAY, 1.0)
    alpert_ratio = energy_deficit_kcal / alpert_capacity
    alpert_stress = 1.0 + 0.35 * clamp(alpert_ratio - 1.0, 0.0, 1.5)

    # Better training and protein reduce, but do not eliminate, FFM loss.
    preservation_modifier = 1.15 - 0.55 * protection
    fraction = clamp(base * preservation_modifier * severity * alpert_stress, 0.02, 0.55)
    return fraction, weekly_rate, alpert_ratio


def _bulk_ffm_fraction(
    fat_mass_kg: float,
    training_quality: float,
    protein_g_per_kg: float,
) -> float:
    base = forbes_ffm_weight_fraction(fat_mass_kg)
    protein = _protein_score(protein_g_per_kg, "Bulk")
    return clamp(base * training_quality * protein, 0.03, 0.55)


def _risk_label(phase: str, weekly_rate: float, alpert_ratio: float) -> str:
    if phase != "Cut":
        return "Low"
    if weekly_rate >= 0.0125 or alpert_ratio >= 1.25:
        return "High"
    if weekly_rate >= 0.008 or alpert_ratio >= 0.90:
        return "Moderate"
    return "Low"


def calculate_projection(
    start_weight: float,
    start_bf: float,
    training_quality: float,
    base_bulk_weeks: float,
    base_cut_weeks: float,
    surplus: int,
    deficit: int,
    total_weeks: float,
    scale_coeff: float,
    start_mode: str,
    first_phase_weeks: float,
    training_status: str = "Intermediate",
    protein_g_per_kg: float = 1.8,
    activity_level: str = "Moderate",
    measured_maintenance_kcal: float | None = None,
    fixed_intake: bool = True,
) -> pd.DataFrame:
    """Calculate a daily body-composition projection.

    The function keeps the original positional API while adding evidence-based
    optional parameters. Output columns retain the old names for compatibility.
    """
    inputs = ProjectionInputs(
        start_weight_kg=float(start_weight),
        start_body_fat_pct=float(start_bf),
        training_quality=float(training_quality),
        training_status=training_status,
        protein_g_per_kg=float(protein_g_per_kg),
        bulk_weeks=float(base_bulk_weeks),
        cut_weeks=float(base_cut_weeks),
        surplus_kcal=int(surplus),
        deficit_kcal=int(deficit),
        total_weeks=float(total_weeks),
        cycle_scale=float(scale_coeff),
        start_mode=start_mode,
        first_phase_weeks=float(first_phase_weeks),
        activity_level=activity_level,
        measured_maintenance_kcal=(
            float(measured_maintenance_kcal)
            if measured_maintenance_kcal is not None
            else None
        ),
        fixed_intake=bool(fixed_intake),
    )
    validate_inputs(inputs)

    total_days = max(1, round(inputs.total_weeks * 7))
    phases = _build_phases(
        total_days,
        inputs.start_mode,
        inputs.first_phase_weeks,
        inputs.bulk_weeks,
        inputs.cut_weeks,
        inputs.cycle_scale,
    )

    fat_mass = inputs.start_weight_kg * inputs.start_body_fat_pct / 100.0
    lean_mass = inputs.start_weight_kg - fat_mass
    initial_rmr = cunningham_rmr_kcal(lean_mass)
    estimated_maintenance = estimate_maintenance_kcal(
        inputs.start_weight_kg,
        inputs.start_body_fat_pct,
        inputs.activity_level,
    )
    baseline_maintenance = (
        inputs.measured_maintenance_kcal
        if inputs.measured_maintenance_kcal is not None
        else estimated_maintenance
    )
    if baseline_maintenance < initial_rmr:
        raise ValueError("Maintenance calories cannot be below estimated resting needs.")

    baseline_non_rmr = baseline_maintenance - initial_rmr
    adaptive_thermogenesis = 0.0
    rows: list[dict[str, float | str | bool]] = []

    def append_row(
        day: int,
        phase: str,
        intake: float,
        tdee: float,
        rmr: float,
        energy_balance: float,
        p_ratio: float,
        weekly_rate: float,
        alpert_ratio: float,
        risk: str,
    ) -> None:
        weight = lean_mass + fat_mass
        rows.append(
            {
                "Day": day,
                "Week": day / 7.0,
                "Weight": weight,
                "BodyFat": 100.0 * fat_mass / weight,
                "LeanMass": lean_mass,
                "FatMass": fat_mass,
                "Phase": phase,
                "Risk": risk,
                "Intake": intake,
                "TDEE": tdee,
                "RMR": rmr,
                "EnergyBalance": energy_balance,
                "AdaptiveThermogenesis": adaptive_thermogenesis,
                "PRatio": p_ratio,
                "WeeklyWeightRate": weekly_rate,
                "FatEnergyStress": alpert_ratio,
                "SafeDeficitLimit": fat_mass * ALPERT_KCAL_PER_KG_FAT_PER_DAY,
                "MaintenanceWasMeasured": inputs.measured_maintenance_kcal is not None,
            }
        )

    # Exact baseline row before any projected change.
    append_row(
        day=0,
        phase=phases[0],
        intake=baseline_maintenance,
        tdee=baseline_maintenance,
        rmr=initial_rmr,
        energy_balance=0.0,
        p_ratio=forbes_ffm_weight_fraction(fat_mass),
        weekly_rate=0.0,
        alpert_ratio=0.0,
        risk="Low",
    )

    for day_index, phase in enumerate(phases, start=1):
        weight = lean_mass + fat_mass
        current_rmr = cunningham_rmr_kcal(lean_mass)
        movement_expenditure = baseline_non_rmr * (weight / inputs.start_weight_kg)
        expenditure_without_adaptation = current_rmr + movement_expenditure

        requested_balance = inputs.surplus_kcal if phase == "Bulk" else -inputs.deficit_kcal
        if inputs.fixed_intake:
            intake = baseline_maintenance + requested_balance
        else:
            intake = expenditure_without_adaptation + adaptive_thermogenesis + requested_balance

        adaptation_target = ADAPTATION_FRACTION * (intake - baseline_maintenance)
        adaptive_thermogenesis += (
            adaptation_target - adaptive_thermogenesis
        ) / ADAPTATION_TIME_CONSTANT_DAYS
        tdee = expenditure_without_adaptation + adaptive_thermogenesis
        energy_balance = intake - tdee

        p_ratio = forbes_ffm_weight_fraction(fat_mass)
        weekly_rate = 0.0
        alpert_ratio = 0.0

        if energy_balance < -1e-9:
            deficit_now = abs(energy_balance)
            p_ratio, weekly_rate, alpert_ratio = _cut_ffm_fraction(
                fat_mass,
                weight,
                deficit_now,
                inputs.training_quality,
                inputs.protein_g_per_kg,
            )
            mixed_density = (
                p_ratio * KCAL_PER_KG_FFM_CHANGE
                + (1.0 - p_ratio) * KCAL_PER_KG_FAT_CHANGE
            )
            delta_weight = energy_balance / mixed_density
            delta_lean = p_ratio * delta_weight
            delta_fat = (1.0 - p_ratio) * delta_weight
        elif energy_balance > 1e-9:
            p_ratio = _bulk_ffm_fraction(
                fat_mass,
                inputs.training_quality,
                inputs.protein_g_per_kg,
            )
            mixed_density = (
                p_ratio * KCAL_PER_KG_FFM_CHANGE
                + (1.0 - p_ratio) * KCAL_PER_KG_FAT_CHANGE
            )
            delta_weight = energy_balance / mixed_density
            proposed_lean = p_ratio * delta_weight
            monthly_cap = (
                MAX_FFM_GAIN_FRACTION_PER_MONTH[inputs.training_status] * weight
            )
            daily_lean_cap = monthly_cap / DAYS_PER_MONTH
            delta_lean = min(proposed_lean, daily_lean_cap)
            remaining_energy = energy_balance - delta_lean * KCAL_PER_KG_FFM_CHANGE
            delta_fat = max(0.0, remaining_energy / KCAL_PER_KG_FAT_CHANGE)
        else:
            delta_lean = 0.0
            delta_fat = 0.0

        lean_mass = max(20.0, lean_mass + delta_lean)
        fat_mass = max(0.5, fat_mass + delta_fat)
        risk = _risk_label(phase, weekly_rate, alpert_ratio)

        append_row(
            day=day_index,
            phase=phase,
            intake=intake,
            tdee=tdee,
            rmr=current_rmr,
            energy_balance=energy_balance,
            p_ratio=p_ratio,
            weekly_rate=weekly_rate,
            alpert_ratio=alpert_ratio,
            risk=risk,
        )

    return pd.DataFrame(rows)


def summarize_phases(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate contiguous phases into a compact summary table."""
    if dataframe.empty:
        return pd.DataFrame()

    df = dataframe.copy()
    df["PhaseGroup"] = (df["Phase"] != df["Phase"].shift()).cumsum()
    rows = []
    for _, group in df.groupby("PhaseGroup", sort=True):
        start = group.iloc[0]
        end = group.iloc[-1]
        rows.append(
            {
                "Phase": start["Phase"],
                "Start week": round(float(start["Week"]), 1),
                "End week": round(float(end["Week"]), 1),
                "Weight change (kg)": round(float(end["Weight"] - start["Weight"]), 2),
                "Fat change (kg)": round(float(end["FatMass"] - start["FatMass"]), 2),
                "Lean change (kg)": round(float(end["LeanMass"] - start["LeanMass"]), 2),
                "Highest risk": (
                    "High"
                    if "High" in set(group["Risk"])
                    else "Moderate"
                    if "Moderate" in set(group["Risk"])
                    else "Low"
                ),
            }
        )
    return pd.DataFrame(rows)
