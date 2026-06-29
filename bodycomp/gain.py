from .core import KCAL_PER_KG_FAT_CHANGE, KCAL_PER_KG_FFM_CHANGE, bulk_daily_ffm_cap


def apply_gain(inputs, state, balance, lean, fat):
    weight = lean + fat
    cap = bulk_daily_ffm_cap(weight, inputs.training_status, inputs.training_quality, inputs.protein_g_per_kg, state.cumulative_bulk_days)
    lean_delta = min(cap, balance * 0.35 / KCAL_PER_KG_FFM_CHANGE)
    fat_delta = max(0.0, (balance - lean_delta * KCAL_PER_KG_FFM_CHANGE) / KCAL_PER_KG_FAT_CHANGE)
    lean += lean_delta
    fat += fat_delta
    total = lean_delta + fat_delta
    state.cumulative_bulk_days += 1
    return lean, fat, lean_delta / total if total > 0 else 0.0
