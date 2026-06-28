from dataclasses import replace

import pytest

from bodycomp import SimulationConfig, forbes_ffm_fraction, simulate, simulate_sensitivity


def test_simulation_includes_exact_starting_state():
    cfg = SimulationConfig(start_weight_kg=80, start_body_fat_pct=20, total_weeks=4)
    df = simulate(cfg)
    assert df.iloc[0]["Weight"] == pytest.approx(80)
    assert df.iloc[0]["BodyFat"] == pytest.approx(20)
    assert len(df) == 4 * 7 + 1


def test_forbes_ffm_fraction_falls_as_fat_mass_rises():
    assert forbes_ffm_fraction(5) > forbes_ffm_fraction(20)


def test_higher_surplus_adds_more_fat_with_same_ffm_ceiling():
    base = SimulationConfig(start_phase="Bulk", total_weeks=8, bulk_weeks=8)
    low = simulate(replace(base, bulk_surplus_kcal_day=100))
    high = simulate(replace(base, bulk_surplus_kcal_day=500))
    assert high.iloc[-1]["FatMass"] > low.iloc[-1]["FatMass"]


def test_more_cut_protection_preserves_more_ffm():
    base = SimulationConfig(start_phase="Cut", total_weeks=8, cut_weeks=8)
    low = simulate(replace(base, lean_loss_protection=0.2))
    high = simulate(replace(base, lean_loss_protection=0.8))
    assert high.iloc[-1]["FatFreeMass"] > low.iloc[-1]["FatFreeMass"]


def test_zero_energy_imbalance_keeps_mass_stable_in_active_phases():
    cfg = SimulationConfig(
        total_weeks=8,
        start_phase="Bulk",
        bulk_weeks=4,
        cut_weeks=4,
        bulk_surplus_kcal_day=0,
        cut_deficit_kcal_day=0,
    )
    df = simulate(cfg)
    assert df.iloc[-1]["Weight"] == pytest.approx(df.iloc[0]["Weight"])


def test_sensitivity_band_contains_base_trajectory():
    base, band = simulate_sensitivity(SimulationConfig(total_weeks=8))
    merged = base.merge(band, on=["Day", "Week"])
    for metric in ["Weight", "BodyFat", "FatMass", "FatFreeMass"]:
        assert (merged[metric] >= merged[f"{metric}_min"] - 1e-9).all()
        assert (merged[metric] <= merged[f"{metric}_max"] + 1e-9).all()


def test_aggressive_cut_uses_user_planning_threshold():
    cfg = SimulationConfig(
        start_phase="Cut",
        total_weeks=2,
        cut_deficit_kcal_day=1200,
        max_weekly_loss_pct=0.5,
    )
    assert simulate(cfg)["AggressiveCut"].any()


def test_invalid_config_fails_early():
    with pytest.raises(ValueError):
        SimulationConfig(start_body_fat_pct=0)
