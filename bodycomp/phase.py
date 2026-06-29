from dataclasses import dataclass

from .core import ProjectionInputs, estimated_days_to_cut_target


@dataclass
class PhaseState:
    phase: str
    previous_phase: str
    days_in_phase: int = 0
    cumulative_bulk_days: int = 0
    final_cut_started: bool = False
    phase_intake_target: float = 0.0


def choose_phase(
    inputs: ProjectionInputs,
    state: PhaseState,
    fixed_phase: str | None,
    remaining_days: int,
    body_fat_pct: float,
    lean_mass: float,
    fat_mass: float,
) -> str:
    if fixed_phase is not None:
        return fixed_phase

    minimum_days = round(inputs.minimum_phase_weeks * 7)
    if inputs.finish_lean and not state.final_cut_started:
        cut_days = estimated_days_to_cut_target(
            lean_mass,
            fat_mass,
            inputs.cut_stop_body_fat_pct,
            inputs.deficit_kcal,
        )
        if remaining_days <= cut_days + 7:
            state.final_cut_started = True
            state.days_in_phase = 0
            return "Cut"

    if state.final_cut_started:
        if state.phase == "Maintain":
            return "Maintain"
        if body_fat_pct <= inputs.cut_stop_body_fat_pct and state.days_in_phase >= 7:
            return "Maintain"
        return "Cut"

    if state.days_in_phase >= minimum_days:
        if state.phase == "Bulk" and body_fat_pct >= inputs.bulk_stop_body_fat_pct:
            state.days_in_phase = 0
            return "Cut"
        if state.phase == "Cut" and body_fat_pct <= inputs.cut_stop_body_fat_pct:
            state.days_in_phase = 0
            return "Bulk"
    return state.phase


def requested_balance(inputs: ProjectionInputs, phase: str) -> float:
    if phase == "Bulk":
        return float(inputs.surplus_kcal)
    if phase == "Cut":
        return -float(inputs.deficit_kcal)
    return 0.0
