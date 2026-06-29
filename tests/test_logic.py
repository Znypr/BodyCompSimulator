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
        starting_transient_state="Neutral / maintenance",
        custom_start_transient_kg=0.0,
    )
    values.update(overrides)
    return logic.calculate_projection(**values)


def test_baseline_matches_entered_scale_weight():
    first = project().iloc[0]
    assert first["Weight"] == pytest.approx(83.0)
    assert first["BodyFat"] == pytest.approx(20.0)
    assert first["ScaleTransient"] == 0.0


def test_neutral_start_includes_initial_cut_transient():
    df = project()
    loss = df.iloc[0]["Weight"] - df.iloc[-1]["Weight"]
    fat_loss = df.iloc[0]["FatMass"] - df.iloc[-1]["FatMass"]
    ffm_loss = df.iloc[0]["LeanMass"] - df.iloc[-1]["LeanMass"]
    assert 5.0 <= loss <= 7.5
    assert 3.5 <= fat_loss <= 5.5
    assert 0.0 <= ffm_loss <= 0.7
    assert df.iloc[-1]["ScaleTransient"] < -1.0


def test_mid_cut_start_does_not_apply_water_loss_twice():
    df = project(
        start_weight=74.0,
        start_bf=12.0,
        total_weeks=30.0 / 7.0,
        first_phase_weeks=30.0 / 7.0,
        starting_transient_state="Already depleted / mid-cut",
    )
    scale_loss = df.iloc[0]["Weight"] - df.iloc[-1]["Weight"]
    transient_change = df.iloc[-1]["ScaleTransient"] - df.iloc[0]["ScaleTransient"]
    assert 1.2 <= scale_loss <= 2.2
    assert abs(transient_change) < 0.15


def test_neutral_start_loses_more_scale_than_mid_cut_start():
    common = dict(
        start_weight=74.0,
        start_bf=12.0,
        total_weeks=30.0 / 7.0,
        first_phase_weeks=30.0 / 7.0,
    )
    neutral = project(starting_transient_state="Neutral / maintenance", **common)
    depleted = project(starting_transient_state="Already depleted / mid-cut", **common)
    neutral_loss = neutral.iloc[0]["Weight"] - neutral.iloc[-1]["Weight"]
    depleted_loss = depleted.iloc[0]["Weight"] - depleted.iloc[-1]["Weight"]
    assert neutral_loss > depleted_loss + 1.0


def test_custom_starting_offset_preserves_entered_scale_weight():
    first = project(
        start_weight=74.0,
        starting_transient_state="Custom",
        custom_start_transient_kg=-1.4,
    ).iloc[0]
    assert first["Weight"] == pytest.approx(74.0)
    assert first["ScaleTransient"] == pytest.approx(-1.4)
    assert first["TissueWeight"] == pytest.approx(75.4)


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


def test_disabling_transients_shows_tissue_loss_only():
    df = project(
        start_weight=74.0,
        start_bf=12.0,
        total_weeks=30.0 / 7.0,
        first_phase_weeks=30.0 / 7.0,
        include_scale_transients=False,
        starting_transient_state="Already depleted / mid-cut",
    )
    assert df.iloc[0]["ScaleTransient"] == pytest.approx(0.0)
    assert df.iloc[-1]["ScaleTransient"] == pytest.approx(0.0)
    assert 1.2 <= df.iloc[0]["Weight"] - df.iloc[-1]["Weight"] <= 2.2


def test_phase_summary_reports_transient_separately():
    summary = logic.summarize_phases(project())
    assert "Stable FFM change (kg)" in summary.columns
    assert "Transient change (kg)" in summary.columns


def test_five_month_500_kcal_gap_projects_about_ten_kg_scale_loss():
    weeks = 5.0 * 365.2425 / 12.0 / 7.0
    df = project(
        start_bf=18.0,
        total_weeks=weeks,
        first_phase_weeks=weeks,
        fixed_intake=False,
        include_scale_transients=True,
        starting_transient_state="Neutral / maintenance",
        activity_level="Moderate",
    )
    cut_rows = df[(df["Phase"] == "Cut") & (df["Day"] > 0)]
    scale_loss = df.iloc[0]["Weight"] - df.iloc[-1]["Weight"]
    tissue_loss = df.iloc[0]["TissueWeight"] - df.iloc[-1]["TissueWeight"]
    average_gap = -cut_rows["EnergyBalance"].mean()

    assert 9.5 <= scale_loss <= 10.7
    assert 7.8 <= tissue_loss <= 8.7
    assert 495 <= average_gap <= 505


def test_fixed_calorie_mode_shrinks_the_cut_gap_over_time():
    weeks = 5.0 * 365.2425 / 12.0 / 7.0
    fixed = project(
        start_bf=18.0,
        total_weeks=weeks,
        first_phase_weeks=weeks,
        fixed_intake=True,
        include_scale_transients=False,
        activity_level="Moderate",
    )
    maintained = project(
        start_bf=18.0,
        total_weeks=weeks,
        first_phase_weeks=weeks,
        fixed_intake=False,
        include_scale_transients=False,
        activity_level="Moderate",
    )
    cut_rows = fixed[(fixed["Phase"] == "Cut") & (fixed["Day"] > 0)]
    first_gap = -cut_rows.iloc[0]["EnergyBalance"]
    final_gap = -cut_rows.iloc[-1]["EnergyBalance"]
    fixed_loss = fixed.iloc[0]["Weight"] - fixed.iloc[-1]["Weight"]
    maintained_loss = maintained.iloc[0]["Weight"] - maintained.iloc[-1]["Weight"]

    assert 490 <= first_gap <= 505
    assert final_gap < 400
    assert fixed_loss < maintained_loss - 1.0
