from .core import ALPERT_KCAL_PER_KG_FAT_PER_DAY


def projection_row(
    day,
    lean,
    fat,
    phase,
    phase_day,
    intake,
    tdee,
    rmr,
    balance,
    adaptation,
    fraction,
    rate,
    stress,
    risk,
    measured,
    final_cut,
):
    weight = lean + fat
    return {
        "Day": day,
        "Week": day / 7.0,
        "Weight": weight,
        "BodyFat": 100.0 * fat / weight,
        "LeanMass": lean,
        "FatMass": fat,
        "Phase": phase,
        "PhaseDay": phase_day,
        "Risk": risk,
        "Intake": intake,
        "TDEE": tdee,
        "RMR": rmr,
        "EnergyBalance": balance,
        "AdaptiveThermogenesis": adaptation,
        "PRatio": fraction,
        "WeeklyWeightRate": rate,
        "FatEnergyStress": stress,
        "SafeDeficitLimit": fat * ALPERT_KCAL_PER_KG_FAT_PER_DAY,
        "MaintenanceWasMeasured": measured,
        "FinalCut": final_cut,
    }
