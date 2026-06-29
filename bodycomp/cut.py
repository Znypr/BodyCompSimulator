from .core import KCAL_PER_KG_FAT_CHANGE, KCAL_PER_KG_FFM_CHANGE, cut_ffm_fraction, risk_label


def apply_cut(inputs, energy_balance, lean_mass, fat_mass):
    weight = lean_mass + fat_mass
    body_fat_pct = 100.0 * fat_mass / weight
    fraction, weekly_rate, stress = cut_ffm_fraction(
        body_fat_pct,
        fat_mass,
        weight,
        abs(energy_balance),
        inputs.training_quality,
        inputs.protein_g_per_kg,
    )
    density = fraction * KCAL_PER_KG_FFM_CHANGE + (1.0 - fraction) * KCAL_PER_KG_FAT_CHANGE
    delta_weight = energy_balance / density
    lean_mass = max(20.0, lean_mass + fraction * delta_weight)
    fat_mass = max(0.5, fat_mass + (1.0 - fraction) * delta_weight)
    return lean_mass, fat_mass, fraction, weekly_rate, stress, risk_label("Cut", weekly_rate, stress)
