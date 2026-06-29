import pandas as pd

from .core import (
    ADAPTATION_FRACTION,
    ADAPTATION_TIME_CONSTANT_DAYS,
    KCAL_PER_KG_FAT_CHANGE,
    build_fixed_phases,
    cunningham_rmr_kcal,
    estimate_maintenance_kcal,
    validate_inputs,
)
from .cut import apply_cut
from .gain import apply_gain
from .phase import PhaseState, choose_phase, requested_balance
from .rows import projection_row


def run_projection(inputs):
    validate_inputs(inputs)
    total_days = max(1, round(inputs.total_weeks * 7))
    fixed = None
    if inputs.cycle_strategy == "Fixed duration":
        fixed = build_fixed_phases(
            total_days,
            inputs.start_mode,
            inputs.first_phase_weeks,
            inputs.bulk_weeks,
            inputs.cut_weeks,
            inputs.cycle_scale,
        )

    fat = inputs.start_weight_kg * inputs.start_body_fat_pct / 100.0
    lean = inputs.start_weight_kg - fat
    initial_rmr = cunningham_rmr_kcal(lean)
    estimated = estimate_maintenance_kcal(
        inputs.start_weight_kg,
        inputs.start_body_fat_pct,
        inputs.activity_level,
    )
    baseline = inputs.measured_maintenance_kcal or estimated
    if baseline < initial_rmr:
        raise ValueError("Maintenance calories cannot be below estimated resting needs.")

    non_rmr = baseline - initial_rmr
    adaptation = 0.0
    state = PhaseState(inputs.start_mode, inputs.start_mode)
    state.phase_intake_target = baseline + requested_balance(inputs, state.phase)
    measured = inputs.measured_maintenance_kcal is not None
    rows = [projection_row(
        0, lean, fat, state.phase, 0, baseline, baseline, initial_rmr,
        0.0, adaptation, 0.0, 0.0, 0.0, "Low", measured, False,
    )]

    for day in range(1, total_days + 1):
        weight = lean + fat
        bf = 100.0 * fat / weight
        rmr = cunningham_rmr_kcal(lean)
        base_tdee = rmr + non_rmr * weight / inputs.start_weight_kg
        current_tdee = base_tdee + adaptation
        fixed_phase = fixed[day - 1] if fixed is not None else None
        next_phase = choose_phase(
            inputs, state, fixed_phase, total_days - day + 1, bf, lean, fat
        )
        changed = next_phase != state.phase
        state.previous_phase = state.phase
        state.phase = next_phase
        if changed:
            state.days_in_phase = 0

        target = requested_balance(inputs, state.phase)
        if changed or day == 1:
            state.phase_intake_target = current_tdee + target
        intake = state.phase_intake_target if inputs.fixed_intake else current_tdee + target

        adaptation_target = ADAPTATION_FRACTION * (intake - baseline)
        adaptation += (adaptation_target - adaptation) / ADAPTATION_TIME_CONSTANT_DAYS
        tdee = base_tdee + adaptation
        balance = intake - tdee
        fraction = rate = stress = 0.0
        risk = "Low"

        if balance < -1e-9:
            lean, fat, fraction, rate, stress, risk = apply_cut(
                inputs, balance, lean, fat
            )
        elif balance > 1e-9 and state.phase == "Bulk":
            lean, fat, fraction = apply_gain(inputs, state, balance, lean, fat)
        elif abs(balance) > 1e-9:
            fat = max(0.5, fat + balance / KCAL_PER_KG_FAT_CHANGE)

        state.days_in_phase += 1
        rows.append(projection_row(
            day, lean, fat, state.phase, state.days_in_phase, intake, tdee, rmr,
            balance, adaptation, fraction, rate, stress, risk, measured,
            state.final_cut_started,
        ))

    return pd.DataFrame(rows)
