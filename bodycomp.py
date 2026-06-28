"""Evidence-informed body-composition scenario model.

The module intentionally exposes assumptions rather than presenting them as
physiological laws. It models fat mass (FM) and fat-free mass (FFM), not
skeletal muscle directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Literal

import pandas as pd

Phase = Literal["Bulk", "Cut", "Maintenance"]

DAYS_PER_WEEK = 7
DAYS_PER_MONTH = 365.25 / 12
FORBES_CONSTANT_KG = 10.4
FAT_ENERGY_DENSITY_KCAL_PER_KG = 9_400.0
FFM_ENERGY_DENSITY_KCAL_PER_KG = 1_800.0


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    start_weight_kg: float = 79.0
    start_body_fat_pct: float = 18.0
    total_weeks: int = 104
    start_phase: Literal["Bulk", "Cut"] = "Cut"
    bulk_weeks: int = 12
    cut_weeks: int = 8
    maintenance_weeks: int = 0
    bulk_surplus_kcal_day: float = 200.0
    cut_deficit_kcal_day: float = 500.0
    lean_gain_rate_pct_bw_month: float = 0.4
    protein_g_per_kg: float = 1.8
    training_adherence: float = 0.9
    lean_loss_protection: float = 0.6
    max_weekly_loss_pct: float = 0.75

    def __post_init__(self) -> None:
        if not 30 <= self.start_weight_kg <= 300:
            raise ValueError("start_weight_kg must be between 30 and 300")
        if not 2 <= self.start_body_fat_pct <= 60:
            raise ValueError("start_body_fat_pct must be between 2 and 60")
        if self.total_weeks < 1:
            raise ValueError("total_weeks must be positive")
        if self.bulk_weeks < 1 or self.cut_weeks < 1:
            raise ValueError("bulk_weeks and cut_weeks must be positive")
        if self.maintenance_weeks < 0:
            raise ValueError("maintenance_weeks cannot be negative")
        if self.bulk_surplus_kcal_day < 0 or self.cut_deficit_kcal_day < 0:
            raise ValueError("energy imbalance values cannot be negative")
        if not 0 <= self.lean_gain_rate_pct_bw_month <= 3:
            raise ValueError("lean gain rate must be between 0 and 3% BW/month")
        if not 0.5 <= self.protein_g_per_kg <= 4:
            raise ValueError("protein_g_per_kg must be between 0.5 and 4")
        if not 0 <= self.training_adherence <= 1:
            raise ValueError("training_adherence must be between 0 and 1")
        if not 0 <= self.lean_loss_protection <= 0.95:
            raise ValueError("lean_loss_protection must be between 0 and 0.95")
        if not 0.1 <= self.max_weekly_loss_pct <= 2.5:
            raise ValueError("max_weekly_loss_pct must be between 0.1 and 2.5")


def forbes_ffm_fraction(fat_mass_kg: float) -> float:
    """Population-level baseline fraction of weight change assigned to FFM.

    This is the differential Forbes/Hall relation. It is used only as a
    starting point for *weight loss*. It is not interpreted as muscle gain,
    anabolic efficiency, or a universal individual prediction.
    """

    fat_mass_kg = max(float(fat_mass_kg), 0.1)
    return FORBES_CONSTANT_KG / (FORBES_CONSTANT_KG + fat_mass_kg)


def protein_adequacy(protein_g_per_kg: float) -> float:
    """Bounded support factor for lean-tissue gain.

    The 1.6 g/kg denominator is an evidence-informed population reference,
    not an individual threshold. The factor is intentionally capped at 1.
    """

    return max(0.0, min(1.0, protein_g_per_kg / 1.6))


def build_schedule(config: SimulationConfig) -> list[Phase]:
    total_days = config.total_weeks * DAYS_PER_WEEK
    active: Phase = config.start_phase
    schedule: list[Phase] = []

    while len(schedule) < total_days:
        active_days = (
            config.bulk_weeks if active == "Bulk" else config.cut_weeks
        ) * DAYS_PER_WEEK
        schedule.extend([active] * active_days)

        if config.maintenance_weeks:
            schedule.extend(
                ["Maintenance"] * config.maintenance_weeks * DAYS_PER_WEEK
            )

        active = "Cut" if active == "Bulk" else "Bulk"

    return schedule[:total_days]


def _bulk_step(
    config: SimulationConfig,
    weight_kg: float,
    fat_mass_kg: float,
    ffm_kg: float,
) -> tuple[float, float]:
    potential_ffm_gain = (
        weight_kg
        * (config.lean_gain_rate_pct_bw_month / 100.0)
        / DAYS_PER_MONTH
        * config.training_adherence
        * protein_adequacy(config.protein_g_per_kg)
    )

    energy_limited_ffm_gain = (
        config.bulk_surplus_kcal_day / FFM_ENERGY_DENSITY_KCAL_PER_KG
        if config.bulk_surplus_kcal_day > 0
        else 0.0
    )
    ffm_gain = min(potential_ffm_gain, energy_limited_ffm_gain)
    remaining_energy = max(
        0.0,
        config.bulk_surplus_kcal_day
        - ffm_gain * FFM_ENERGY_DENSITY_KCAL_PER_KG,
    )
    fat_gain = remaining_energy / FAT_ENERGY_DENSITY_KCAL_PER_KG
    return fat_mass_kg + fat_gain, ffm_kg + ffm_gain


def _cut_step(
    config: SimulationConfig,
    weight_kg: float,
    fat_mass_kg: float,
    ffm_kg: float,
) -> tuple[float, float, float, float]:
    baseline_ffm_fraction = forbes_ffm_fraction(fat_mass_kg)
    adjusted_ffm_fraction = baseline_ffm_fraction * (
        1.0 - config.lean_loss_protection
    )
    adjusted_ffm_fraction = max(0.0, min(0.95, adjusted_ffm_fraction))

    effective_energy_density = (
        adjusted_ffm_fraction * FFM_ENERGY_DENSITY_KCAL_PER_KG
        + (1.0 - adjusted_ffm_fraction) * FAT_ENERGY_DENSITY_KCAL_PER_KG
    )
    weight_loss = (
        config.cut_deficit_kcal_day / effective_energy_density
        if effective_energy_density > 0
        else 0.0
    )

    ffm_loss = min(weight_loss * adjusted_ffm_fraction, max(0.0, ffm_kg - 20.0))
    fat_loss = min(
        weight_loss * (1.0 - adjusted_ffm_fraction),
        max(0.0, fat_mass_kg - 0.1),
    )

    next_fat_mass = fat_mass_kg - fat_loss
    next_ffm = ffm_kg - ffm_loss
    weekly_loss_pct = (
        ((fat_loss + ffm_loss) * DAYS_PER_WEEK / weight_kg) * 100.0
        if weight_kg > 0
        else 0.0
    )
    return next_fat_mass, next_ffm, adjusted_ffm_fraction, weekly_loss_pct


def simulate(config: SimulationConfig) -> pd.DataFrame:
    """Run a deterministic scenario and include the exact starting state."""

    fat_mass = config.start_weight_kg * config.start_body_fat_pct / 100.0
    ffm = config.start_weight_kg - fat_mass
    schedule = build_schedule(config)

    rows: list[dict[str, float | int | str | bool]] = []

    def append_row(
        day: int,
        phase: Phase | Literal["Start"],
        adjusted_ffm_fraction: float = 0.0,
        weekly_loss_pct: float = 0.0,
    ) -> None:
        weight = fat_mass + ffm
        rows.append(
            {
                "Day": day,
                "Week": day / DAYS_PER_WEEK,
                "Weight": weight,
                "BodyFat": fat_mass / weight * 100.0,
                "FatMass": fat_mass,
                "FatFreeMass": ffm,
                "Phase": phase,
                "ForbesFFMFraction": forbes_ffm_fraction(fat_mass),
                "AdjustedFFMFraction": adjusted_ffm_fraction,
                "WeeklyLossPct": weekly_loss_pct,
                "AggressiveCut": weekly_loss_pct > config.max_weekly_loss_pct,
            }
        )

    append_row(0, "Start")

    for day, phase in enumerate(schedule, start=1):
        weight = fat_mass + ffm
        adjusted_ffm_fraction = 0.0
        weekly_loss_pct = 0.0

        if phase == "Bulk":
            fat_mass, ffm = _bulk_step(config, weight, fat_mass, ffm)
        elif phase == "Cut":
            fat_mass, ffm, adjusted_ffm_fraction, weekly_loss_pct = _cut_step(
                config, weight, fat_mass, ffm
            )

        fat_mass = max(0.1, fat_mass)
        ffm = max(20.0, ffm)
        append_row(day, phase, adjusted_ffm_fraction, weekly_loss_pct)

    return pd.DataFrame(rows)


def simulate_sensitivity(
    config: SimulationConfig,
    body_fat_error_pp: float = 2.0,
    response_error_fraction: float = 0.25,
    protection_error: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return base simulation and a user-defined sensitivity envelope.

    The envelope is not a confidence interval. It shows how output changes
    when the starting BF estimate and two response assumptions are varied.
    """

    if body_fat_error_pp < 0 or response_error_fraction < 0 or protection_error < 0:
        raise ValueError("sensitivity values cannot be negative")

    base = simulate(config)
    variants: list[pd.DataFrame] = []

    bf_values = {
        max(2.0, config.start_body_fat_pct - body_fat_error_pp),
        config.start_body_fat_pct,
        min(60.0, config.start_body_fat_pct + body_fat_error_pp),
    }
    rate_values = {
        max(0.0, config.lean_gain_rate_pct_bw_month * (1 - response_error_fraction)),
        config.lean_gain_rate_pct_bw_month,
        min(3.0, config.lean_gain_rate_pct_bw_month * (1 + response_error_fraction)),
    }
    protection_values = {
        max(0.0, config.lean_loss_protection - protection_error),
        config.lean_loss_protection,
        min(0.95, config.lean_loss_protection + protection_error),
    }

    for bf, rate, protection in product(bf_values, rate_values, protection_values):
        variant = replace(
            config,
            start_body_fat_pct=bf,
            lean_gain_rate_pct_bw_month=rate,
            lean_loss_protection=protection,
        )
        variants.append(simulate(variant))

    combined = pd.concat(variants, ignore_index=True)
    metrics = ["Weight", "BodyFat", "FatMass", "FatFreeMass"]
    grouped = combined.groupby("Day", sort=True)[metrics].agg(["min", "max"])
    grouped.columns = [f"{metric}_{bound}" for metric, bound in grouped.columns]
    band = grouped.reset_index()
    band["Week"] = band["Day"] / DAYS_PER_WEEK
    return base, band
