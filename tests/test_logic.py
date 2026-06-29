import pytest

import logic


def project(**overrides):
    values = dict(
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
        cycle_strategy="Fixed duration",
        bulk_stop_body_fat_pct=15.0,
        cut_stop_body_fat_pct=10.0,
        minimum_phase_weeks=4.0,
        finish_lean=False,
    )
    values.update(overrides)
    return logic.calculate_projection(**values)


def test_baseline_matches_inputs():
    first = project().iloc[0]
    assert first["Weight"] == pytest.approx(80.0)
    assert first["BodyFat"] == pytest.approx(20.0)
    assert first["FatMass"] == pytest.approx(16.0)
    assert first["LeanMass"] == pytest.approx(64.0)


def test_zero_balance_is_stable():
    df = project(surplus=0, start_mode="Bulk", first_phase_weeks=12)
    assert df.iloc[-1]["Weight"] == pytest.approx(df.iloc[0]["Weight"], abs=0.03)


def test_new_cut_is_anchored_to_current_maintenance():
    df = project(
        start_mode="Bulk",
        first_phase_weeks=8,
        total_weeks=16,
        surplus=250,
        deficit=500,
    )
    first_cut = df[df["Phase"] == "Cut"].iloc[0]
    assert first_cut["EnergyBalance"] == pytest.approx(-500, abs=15)


def test_cut_loses_mostly_fat_with_high_protection():
    df = project(deficit=500, protein_g_per_kg=2.2, training_quality=0.95)
    fat_loss = df.iloc[0]["FatMass"] - df.iloc[-1]["FatMass"]
    ffm_loss = df.iloc[0]["LeanMass"] - df.iloc[-1]["LeanMass"]
    assert fat_loss > 0
    assert ffm_loss >= 0
    assert fat_loss > 8 * ffm_loss


def test_training_and_protein_preserve_more_ffm():
    protected = project(training_quality=1.0, protein_g_per_kg=2.4)
    unprotected = project(training_quality=0.2, protein_g_per_kg=0.8)
    protected_loss = protected.iloc[0]["LeanMass"] - protected.iloc[-1]["LeanMass"]
    unprotected_loss = unprotected.iloc[0]["LeanMass"] - unprotected.iloc[-1]["LeanMass"]
    assert protected_loss < unprotected_loss


def test_beginner_gain_exceeds_advanced_gain():
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
    assert beginner_gain > advanced_gain > 0


def test_body_fat_range_cycle_improves_lean_checkpoint():
    df = project(
        start_weight=74,
        start_bf=11,
        start_mode="Bulk",
        total_weeks=104,
        cycle_strategy="Body-fat range",
        bulk_stop_body_fat_pct=15,
        cut_stop_body_fat_pct=8,
        finish_lean=True,
        training_status="Advanced",
        training_quality=1.0,
        protein_g_per_kg=2.2,
        measured_maintenance_kcal=2339,
    )
    assert "Bulk" in set(df["Phase"])
    assert "Cut" in set(df["Phase"])
    assert df.iloc[-1]["LeanMass"] > df.iloc[0]["LeanMass"]
    assert df.iloc[-1]["BodyFat"] <= df.iloc[0]["BodyFat"]


def test_aggressive_cut_increases_risk_and_ffm_loss():
    moderate = project(deficit=900)
    aggressive = project(deficit=1100)
    moderate_loss = moderate.iloc[0]["LeanMass"] - moderate.iloc[-1]["LeanMass"]
    aggressive_loss = aggressive.iloc[0]["LeanMass"] - aggressive.iloc[-1]["LeanMass"]
    assert aggressive_loss > moderate_loss
    assert set(aggressive["Risk"]).issubset({"Low", "Moderate", "High"})


def test_forbes_reference_decreases_with_fat_mass():
    assert logic.forbes_ffm_weight_fraction(8) > logic.forbes_ffm_weight_fraction(20)


def test_summary_uses_metric_labels():
    summary = logic.summarize_phases(project())
    assert "End weight (kg)" in summary.columns
    assert "Fat change (kg)" in summary.columns
    assert "Fat-free change (kg)" in summary.columns
