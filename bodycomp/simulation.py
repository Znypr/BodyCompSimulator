from __future__ import annotations

import pandas as pd

from .core import (
    ADAPTATION_TIME_CONSTANT_DAYS,
    BULK_ADAPTATION_FRACTION,
    CUT_ADAPTATION_FRACTION,
    KCAL_PER_KG_FAT_CHANGE,
    KCAL_PER_KG_FFM_CHANGE,
    TRANSIENT_TIME_CONSTANT_DAYS,
    ProjectionInputs,
    build_fixed_phases,
    bulk_ffm_gain_cap_kg_per_day,
    clamp,
    cunningham_rmr_kcal,
    cut_ffm_fraction,
    estimate_bulk_transient_target_kg,
    estimate_cut_transient_target_kg,
    estimate_maintenance_kcal,
    validate_inputs,
)


def _phase_target_transient(inputs: ProjectionInputs, phase: str, stable_weight: float) -> float:
    if not inputs.include_scale_transients:
        return 0.0
    if phase == "Cut":
        return -estimate_cut_transient_target_kg(stable_weight, inputs.deficit_kcal * inputs.cut_gap_multiplier)
    if phase == "Bulk":
        return estimate_bulk_transient_target_kg(stable_weight, inputs.surplus_kcal)
    return 0.0


def _requested_balance(inputs: ProjectionInputs, phase: str) -> float:
    if phase == "Bulk":
        return float(inputs.surplus_kcal)
    if phase == "Cut":
        return -float(inputs.deficit_kcal) * inputs.cut_gap_multiplier
    return 0.0


def _estimated_cut_days(inputs: ProjectionInputs, lean: float, fat: float) -> int:
    target = inputs.cut_stop_body_fat_pct / 100.0
    target_fat = target * lean / max(1.0 - target, 0.01)
    fat_to_lose = max(0.0, fat - target_fat)
    effective_gap = max(inputs.deficit_kcal * inputs.cut_gap_multiplier * 0.85, 1.0)
    return max(7, round(fat_to_lose * KCAL_PER_KG_FAT_CHANGE / effective_gap))


def _body_fat_phase_switch(inputs: ProjectionInputs, phase: str, body_fat_pct: float, days_in_phase: int) -> str:
    if inputs.cycle_strategy != "Body-fat range" or days_in_phase < round(inputs.minimum_phase_weeks * 7):
        return phase
    if phase == "Bulk" and body_fat_pct >= inputs.bulk_stop_body_fat_pct:
        return "Cut"
    if phase == "Cut" and body_fat_pct <= inputs.cut_stop_body_fat_pct:
        return "Bulk"
    return phase


def calculate_projection(
    start_weight,
    start_bf,
    training_quality,
    base_bulk_weeks,
    base_cut_weeks,
    surplus,
    deficit,
    total_weeks,
    scale_coeff,
    start_mode,
    first_phase_weeks,
    training_status="Intermediate",
    protein_g_per_kg=1.8,
    activity_level="Moderate",
    measured_maintenance_kcal=None,
    fixed_intake=True,
    cycle_strategy="Fixed duration",
    bulk_stop_body_fat_pct=15.0,
    cut_stop_body_fat_pct=10.0,
    minimum_phase_weeks=4.0,
    finish_lean=False,
    cut_gap_multiplier=1.0,
    include_scale_transients=True,
):
    inputs = ProjectionInputs(
        float(start_weight), float(start_bf), float(training_quality), training_status,
        float(protein_g_per_kg), float(base_bulk_weeks), float(base_cut_weeks),
        int(surplus), int(deficit), float(total_weeks), float(scale_coeff),
        start_mode, float(first_phase_weeks), activity_level,
        float(measured_maintenance_kcal) if measured_maintenance_kcal is not None else None,
        bool(fixed_intake), cycle_strategy, float(bulk_stop_body_fat_pct),
        float(cut_stop_body_fat_pct), float(minimum_phase_weeks), bool(finish_lean),
        float(cut_gap_multiplier), bool(include_scale_transients),
    )
    validate_inputs(inputs)

    total_days = max(1, round(inputs.total_weeks * 7))
    fixed_phases = None
    if inputs.cycle_strategy == "Fixed duration":
        fixed_phases = build_fixed_phases(total_days, inputs.start_mode, inputs.first_phase_weeks, inputs.bulk_weeks, inputs.cut_weeks, inputs.cycle_scale)

    fat = inputs.start_weight_kg * inputs.start_body_fat_pct / 100.0
    lean = inputs.start_weight_kg - fat
    transient = 0.0
    initial_rmr = cunningham_rmr_kcal(lean)
    estimated_maintenance = estimate_maintenance_kcal(inputs.start_weight_kg, inputs.start_body_fat_pct, inputs.activity_level)
    baseline_maintenance = inputs.measured_maintenance_kcal or estimated_maintenance
    if baseline_maintenance < initial_rmr:
        raise ValueError("Maintenance calories cannot be below estimated resting needs.")

    baseline_non_rmr = baseline_maintenance - initial_rmr
    adaptation = 0.0
    phase = inputs.start_mode
    previous_phase = phase
    days_in_phase = 0
    cumulative_bulk_days = 0
    final_cut_started = False
    phase_intake = baseline_maintenance + _requested_balance(inputs, phase)
    rows = []

    def add_row(day, intake, tdee, balance, ffm_fraction, weekly_rate):
        stable_weight = lean + fat
        scale_weight = stable_weight + transient
        rows.append({
            "Day": day,
            "Week": day / 7.0,
            "Weight": scale_weight,
            "TissueWeight": stable_weight,
            "BodyFat": 100.0 * fat / stable_weight,
            "LeanMass": lean,
            "FatMass": fat,
            "ScaleTransient": transient,
            "Phase": phase,
            "PhaseDay": days_in_phase,
            "Intake": intake,
            "TDEE": tdee,
            "RMR": cunningham_rmr_kcal(lean),
            "EnergyBalance": balance,
            "AdaptiveThermogenesis": adaptation,
            "PRatio": ffm_fraction,
            "WeeklyWeightRate": weekly_rate,
            "Risk": "High" if weekly_rate > 0.012 else "Moderate" if weekly_rate > 0.008 else "Low",
            "FinalCut": final_cut_started,
        })

    add_row(0, baseline_maintenance, baseline_maintenance, 0.0, 0.0, 0.0)

    for day in range(1, total_days + 1):
        stable_weight = lean + fat
        tissue_bf = 100.0 * fat / stable_weight

        if fixed_phases is not None:
            phase = fixed_phases[day - 1]
        else:
            remaining_days = total_days - day + 1
            if inputs.finish_lean and not final_cut_started:
                if remaining_days <= _estimated_cut_days(inputs, lean, fat) + 7:
                    phase = "Cut"
                    final_cut_started = True
            if final_cut_started:
                if phase == "Maintain":
                    phase = "Maintain"
                elif tissue_bf <= inputs.cut_stop_body_fat_pct and days_in_phase >= 7:
                    phase = "Maintain"
                else:
                    phase = "Cut"
            else:
                phase = _body_fat_phase_switch(inputs, phase, tissue_bf, days_in_phase)

        changed = phase != previous_phase
        if changed:
            days_in_phase = 0

        rmr = cunningham_rmr_kcal(lean)
        movement = baseline_non_rmr * (stable_weight / inputs.start_weight_kg)
        tdee_before_adaptation = rmr + movement
        current_tdee = tdee_before_adaptation + adaptation
        requested = _requested_balance(inputs, phase)

        if changed or day == 1:
            phase_intake = current_tdee + requested
        intake = phase_intake if inputs.fixed_intake else current_tdee + requested

        adaptation_fraction = BULK_ADAPTATION_FRACTION if phase == "Bulk" else CUT_ADAPTATION_FRACTION if phase == "Cut" else 0.0
        target_adaptation = adaptation_fraction * (intake - baseline_maintenance)
        adaptation += (target_adaptation - adaptation) / ADAPTATION_TIME_CONSTANT_DAYS
        tdee = tdee_before_adaptation + adaptation
        balance = intake - tdee

        ffm_fraction = 0.0
        weekly_rate = 0.0
        if balance < -1e-9:
            ffm_fraction, weekly_rate = cut_ffm_fraction(tissue_bf, stable_weight, abs(balance), inputs.training_quality, inputs.protein_g_per_kg)
            density = ffm_fraction * KCAL_PER_KG_FFM_CHANGE + (1.0 - ffm_fraction) * KCAL_PER_KG_FAT_CHANGE
            delta_weight = balance / density
            lean += ffm_fraction * delta_weight
            fat += (1.0 - ffm_fraction) * delta_weight
        elif balance > 1e-9 and phase == "Bulk":
            cap = bulk_ffm_gain_cap_kg_per_day(inputs, stable_weight, cumulative_bulk_days)
            potential_lean = min(cap, balance * 0.45 / KCAL_PER_KG_FFM_CHANGE)
            lean_energy = potential_lean * KCAL_PER_KG_FFM_CHANGE
            residual = max(0.0, balance - lean_energy)
            storage_efficiency = clamp(0.55 + 0.20 * (inputs.surplus_kcal / 500.0), 0.50, 0.78)
            fat_gain = residual * storage_efficiency / KCAL_PER_KG_FAT_CHANGE
            lean += potential_lean
            fat += fat_gain
            total_gain = potential_lean + fat_gain
            ffm_fraction = potential_lean / total_gain if total_gain > 0 else 0.0
            cumulative_bulk_days += 1

        lean = max(20.0, lean)
        fat = max(0.5, fat)
        transient_target = _phase_target_transient(inputs, phase, lean + fat)
        transient += (transient_target - transient) / TRANSIENT_TIME_CONSTANT_DAYS

        days_in_phase += 1
        add_row(day, intake, tdee, balance, ffm_fraction, weekly_rate)
        previous_phase = phase

    return pd.DataFrame(rows)
