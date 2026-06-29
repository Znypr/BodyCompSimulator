import logic


def test_final_cut_stays_at_maintenance_after_reaching_target():
    df = logic.calculate_projection(
        start_weight=74,
        start_bf=11,
        training_quality=1.0,
        base_bulk_weeks=20,
        base_cut_weeks=20,
        surplus=250,
        deficit=500,
        total_weeks=74,
        scale_coeff=1.0,
        start_mode="Bulk",
        first_phase_weeks=20,
        training_status="Advanced",
        protein_g_per_kg=2.2,
        activity_level="Sedentary",
        fixed_intake=True,
        cycle_strategy="Body-fat range",
        bulk_stop_body_fat_pct=15,
        cut_stop_body_fat_pct=8,
        minimum_phase_weeks=4,
        finish_lean=True,
    )
    assert df.iloc[-1]["Phase"] == "Maintain"
    assert 7.5 <= df.iloc[-1]["BodyFat"] <= 8.2
