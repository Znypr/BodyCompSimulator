import pytest
import logic


def project(**overrides):
    values = dict(
        start_weight=83.0,
        start_bf=20.0,
        training_quality=0.95,
        base_bulk_weeks=42,
        base_cut_weeks=12,
        surplus=225,
        deficit=500,
        total_weeks=11,
        scale_coeff=1.0,
        start_mode="Cut",
        first_phase_weeks=11,
        training_status="Advanced",
        protein_g_per_kg=2.0,
        activity_level="Light",
        fixed_intake=False,
        cycle_strategy="Fixed duration",
        bulk_stop_body_fat_pct=15.0,
        cut_stop_body_fat_pct=10.0,
        minimum_phase_weeks=4.0,
        finish_lean=False,
        cut_gap_multiplier=1.0,
        include_scale_transients=True,
    )
    values.update(overrides)
    return logic.calculate_projection(**values)


def test_baseline_matches_inputs():
    first = project().iloc[0]
    assert first["Weight"] == pytest.approx(83.0)
    assert first["BodyFat"] == pytest.approx(20.0)
    assert first["ScaleTransient"] == 0.0


def test_true_500_deficit_includes_tissue_and_transient_loss():
    df = project()
    loss = df.iloc[0]["Weight"] - df.iloc[-1]["Weight"]
    fat_loss = df.iloc[0]["FatMass"] - df.iloc[-1]["FatMass"]
    ffm_loss = df.iloc[0]["LeanMass"] - df.iloc[-1]["LeanMass"]
    assert 5.0 <= loss <= 7.5
    assert 3.5 <= fat_loss <= 5.5
    assert 0.0 <= ffm_loss <= 0.7
    assert df.iloc[-1]["ScaleTransient"] < -1.0


def test_observed_cut_calibration_matches_83_to_74_trend():
    calibration = logic.infer_cut_calibration(83, 74, 11, 500)
    assert 700 <= calibration["effective_deficit_kcal"] <= 850
    assert 1.4 <= calibration["cut_gap_multiplier"] <= 1.8
    df = project(cut_gap_multiplier=calibration["cut_gap_multiplier"])
    assert 73.5 <= df.iloc[-1]["Weight"] <= 75.0


def test_transient_is_not_counted_as_stable_ffm_loss():
    df = project()
    stable_ffm_loss = df.iloc[0]["LeanMass"] - df.iloc[-1]["LeanMass"]
    scale_loss = df.iloc[0]["Weight"] - df.iloc[-1]["Weight"]
    assert stable_ffm_loss < 0.5
    assert scale_loss > stable_ffm_loss + 4.0


def test_hypertrophy_cycle_is_not_forced_into_six_kg_fat_gain():
    df = project(
        start_weight=74,
        start_bf=11,
        total_weeks=52,
        start_mode="Bulk",
        first_phase_weeks=42,
        base_bulk_weeks=42,
        base_cut_weeks=10,
        fixed_intake=False,
    )
    fat_change = df.iloc[-1]["FatMass"] - df.iloc[0]["FatMass"]
    ffm_change = df.iloc[-1]["LeanMass"] - df.iloc[0]["LeanMass"]
    assert 1.0 <= ffm_change <= 3.0
    assert -0.5 <= fat_change <= 2.0


def test_without_transients_500_deficit_shows_tissue_loss_only():
    df = project(include_scale_transients=False)
    assert df.iloc[-1]["ScaleTransient"] == pytest.approx(0.0)
    assert 3.5 <= df.iloc[0]["Weight"] - df.iloc[-1]["Weight"] <= 5.5


def test_phase_summary_reports_transient_separately():
    summary = logic.summarize_phases(project())
    assert "Stable FFM change (kg)" in summary.columns
    assert "Transient change (kg)" in summary.columns
