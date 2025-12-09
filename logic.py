import numpy as np
import pandas as pd

def calculate_projection(
    start_weight: float,
    start_bf: float,
    training_quality: float,
    base_bulk_weeks: float,
    base_cut_weeks: float,
    surplus: int,
    deficit: int,
    total_weeks: int,
    scale_coeff: float,
    start_mode: str,
    first_phase_weeks: float,
    bulk_fatigue_factor: float
):
    total_days = int(total_weeks * 7)
    phases = []
    
    # --- 1. Generate The Timeline ---
    first_len = int(first_phase_weeks * 7)
    if start_mode == 'Bulk':
        phases.extend(['Bulk'] * first_len)
        next_is_bulk = False 
    else:
        phases.extend(['Cut'] * first_len)
        next_is_bulk = True

    current_day_count = len(phases)
    cycle_idx = 0
    
    while current_day_count < total_days:
        multiplier = scale_coeff ** cycle_idx
        b_len = int(base_bulk_weeks * 7 * multiplier)
        c_len = int(base_cut_weeks * 7 * multiplier)
        
        if next_is_bulk:
            phases.extend(['Bulk'] * b_len)
            phases.extend(['Cut'] * c_len)
        else:
            phases.extend(['Cut'] * c_len)
            phases.extend(['Bulk'] * b_len)
            
        current_day_count += (b_len + c_len)
        cycle_idx += 1
        
    phases = phases[:total_days]
    
    # --- 2. Simulation Loop ---
    days_data = []
    
    current_fm = start_weight * (start_bf / 100.0)
    current_lm = start_weight - current_fm
    
    # Constants
    kcal_fat = 7700.0      # Energy to burn 1kg fat
    kcal_muscle = 1800.0   # Energy to burn 1kg wet muscle tissue
    
    consecutive_bulk_days = 0
    
    for day_idx in range(total_days):
        phase = phases[day_idx]
        theoretical_p_ratio = 10.4 / (10.4 + current_fm)
        
        is_unsafe_cut = False # Flag for visualization
        daily_max_fat_transfer = 0
        
        if phase == 'Bulk':
            consecutive_bulk_days += 1
            weeks_in_bulk = consecutive_bulk_days / 7.0
            fatigue_penalty = weeks_in_bulk * bulk_fatigue_factor
            current_quality = max(0.1, training_quality - fatigue_penalty)
            
            actual_lean_ratio = theoretical_p_ratio * current_quality
            
            delta_weight = surplus / kcal_fat
            lean_gain = delta_weight * actual_lean_ratio
            fat_gain = delta_weight * (1 - actual_lean_ratio)
            
            current_lm += lean_gain
            current_fm += fat_gain
            
        else:
            consecutive_bulk_days = 0
            
            # --- ALPERT LIMIT LOGIC (The Consequence) ---
            # Max energy fat can release per day
            max_fat_energy = current_fm * 69.0
            daily_max_fat_transfer = max_fat_energy
            
            if deficit <= max_fat_energy:
                # SAFE ZONE: Fat can handle the deficit
                # Efficiency slider determines small protein turnover
                # Standard model: "Efficiency" = Fraction of weight lost that is fat
                delta_weight = deficit / kcal_fat
                fat_loss = delta_weight * training_quality
                lean_loss = delta_weight * (1 - training_quality)
            else:
                # DANGER ZONE: Deficit exceeds fat's transfer rate
                is_unsafe_cut = True
                
                # 1. Take all possible energy from fat
                fat_loss = max_fat_energy / kcal_fat
                
                # 2. Remainder must come from muscle
                remaining_deficit = deficit - max_fat_energy
                # Muscle is less energy dense, so you lose MASS faster here
                lean_loss_penalty = remaining_deficit / kcal_muscle 
                
                # Add baseline lean loss from training imperfections
                # (Even if safe, you lose some. Now you lose that PLUS the penalty)
                base_lean_loss = (max_fat_energy / kcal_fat) * (1 - training_quality)
                
                lean_loss = base_lean_loss + lean_loss_penalty

            current_fm -= fat_loss
            current_lm -= lean_loss

        # Prevent negatives
        current_fm = max(0.1, current_fm)
        current_lm = max(20, current_lm)

        current_w = current_lm + current_fm
        current_bf_p = (current_fm / current_w) * 100.0
        
        # Determine strict phase label for graphing
        # We distinguish "Cut" vs "Unsafe Cut"
        graph_phase = "Unsafe" if is_unsafe_cut else phase
        
        days_data.append({
            "Week": day_idx / 7.0,
            "Weight": current_w,
            "BodyFat": current_bf_p,
            "LeanMass": current_lm,
            "FatMass": current_fm,
            "Phase": graph_phase, # Used for coloring
            "SafeDeficitLimit": daily_max_fat_transfer if phase == "Cut" else 0
        })
        
    return pd.DataFrame(days_data)