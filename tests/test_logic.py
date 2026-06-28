import pytest

import logic


def project(**overrides):
    defaults = dict(
        start_weight=80.0,
        start_bf=20.0,
        training_quality=0.85,
        base_bulk_weeks=12,
        base_cut_weeks=8,
        surplus=250,
        deficit=500,
        total_weeks=12,
        scale_coeff=1.0,
        start_mode="Cut",
        first_phase_weeks=12,
        training_status="Intermediate",
        protein_g_per_kg=2.0,
        activity_level="Moderate",
        measured_maintenance_kcal=2600,
        fixed_intake=True,
    )
    defaults.update(overrides)
    return logic.calculate_projection(**defaults)


def test_baseline_row_matches_user_inputs_exactly():
    df = project()
    first = df.iloc[0]
    assert first["Weight"] == pytest.approx(80.0)
    assert first["BodyFat"] == pytest.approx(20.0)
    assert first["FatMass"] == pytest.approx(16.0)
    assert first["LeanMass"] == pytest.approx(64.0)


def test_zero_energy_balance_is_weight_stable():
    df = project(
        surplus=0,
        start_mode="Bulk",
        first_phase_weeks=12,
        measured_maintenance_kcal=2600,
    )
    assert df.iloc[-1]["Weight"] == pytest.approx(df.iloc[0]["Weight"], abs=0.02)


def test_fixed_intake_deficit_shrinks_as_body_mass_falls():
    df = project(deficit=600)
    early = abs(df.iloc[1]["EnergyBalance"])
    late = abs(df.iloc[-1]["EnergyBalance"])
    assert late < early


def test_cut_reduces_fat_and_only_modestly_reduces_lean_mass():
    df = project(deficit=500, protein_g_per_kg=2.2, training_quality=0.9)
    fat_loss = df.iloc[0]["FatMass"] - df.iloc[-1]["FatMass"]
    lean_loss = df.iloc[0]["LeanMass"] - df.iloc[-1]["LeanMass"]
    assert fat_loss > 0
    assert lean_loss >= 0
    assert fat_loss > lean_loss


def test_training_and_protein_preserve_more_lean_mass_in_cut():
    protected = project(training_quality=1.0, protein_g_per_kg=2.4)
    unprotected = project(training_quality=0.2, protein_g_per_kg=0.8)
    protected_loss = protected.iloc[0]["LeanMass"] - protected.iloc[-1]["LeanMass"]
    unprotected_loss = unprotected.iloc[0]["LeanMass"] - unprotected.iloc[-1]["LeanMass"]
    assert protected_loss < unprotected_loss


def test_advanced_bulk_projects_less_lean_gain_than_beginner():
    common = dict(
        start_mode="Bulk",
        first_phase_weeks=24,
        total_weeks=24,
        surplus=500,
        deficit=0,
        protein_g_per_kg=1.8,
        training_quality=1.0,
    )
    beginner = project(training_status="Beginner", **common)
    advanced = project(training_status="Advanced", **common)
    beginner_gain = beginner.iloc[-1]["LeanMass"] - beginner.iloc[0]["LeanMass"]
    advanced_gain = advanced.iloc[-1]["LeanMass"] - advanced.iloc[0]["LeanMass"]
    assert beginner_gain > advanced_gain


def test_alpert_estimate_is_a_risk_signal_not_a_discontinuous_switch():
    moderate = project(deficit=900)
    aggressive = project(deficit=1100)
    assert aggressive.iloc[-1]["LeanMass"] < moderate.iloc[-1]["LeanMass"]
    assert set(aggressive["Risk"]).issubset({"Low", "Moderate", "High"})
    assert not aggressive["Weight"].isna().any()


def test_forbes_fraction_decreases_with_fat_mass():
    assert logic.forbes_ffm_weight_fraction(8) > logic.forbes_ffm_weight_fraction(20)


def test_phase_summary_uses_metric_units():
    summary = logic.summarize_phases(project())
    assert "Weight change (kg)" in summary.columns
    assert "Fat change (kg)" in summary.columns
    assert "Lean change (kg)" in summary.columns
