import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# logic.py

def get_canonical_p_ratio(bf_percent):
    """
    Solves the Forbes equation for the canonical 'reference' curve.
    Formula: LBM = 10.4 * ln(F) + 14.2
    """
    # 1. Safety Clamp: Forbes equation breaks near 0% BF. 
    if bf_percent <= 1.0: 
        return 0.95
        
    target = bf_percent / 100.0
    
    # Equation: F / (F + 10.4*ln(F) + 14.2) - target = 0
    # We restrict F (Fat Mass) to be positive to avoid log(negative) errors
    def func(f):
        # A low f value will cause log errors; clamp f to a small positive number
        if f < 0.1: return 1.0 
        return (f / (f + 10.4 * np.log(f) + 14.2)) - target
    
    # Determine a smarter initial guess:
    # Use 15kg for high BF, and a lower guess for low BF% to help convergence.
    initial_guess = 15.0
    if bf_percent < 15.0:
        initial_guess = 5.0 # Lower guess for leaner individuals
    
    try:
        f_sol = fsolve(func, initial_guess, maxfev=1000)[0] # Increase max iterations
        
        # If solver returns a non-physical value, apply the high-efficiency clamp
        if f_sol <= 0.1: 
            return 0.95 
        
        # Calculate FFM Proportion of Delta BW: dFFM/dBW = 10.4 / (10.4 + F)
        p_ratio = 10.4 / (10.4 + f_sol)
        return p_ratio
    except ValueError:
        # Handle cases where log(negative) occurs due to solver wandering
        return 0.95 # Assume max efficiency for extreme leanness
    except RuntimeError:
        # Handle non-convergence (The solver did not find a root)
        # This is the most likely cause of the "break"
        if bf_percent < 15.0: 
            return 0.95 # Assume max efficiency
        else:
            # For non-lean states, return a value that keeps the curve smooth
            return 10.4 / (10.4 + 20.0) # Approx 34% FFM proportion for extreme failure

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
    
    kcal_fat = 7700.0
    kcal_muscle = 1800.0
    consecutive_bulk_days = 0
    
    for day_idx in range(total_days):
        phase = phases[day_idx]
        
        # Forbes Calculation (p-ratio) for the specific user
        # p = 10.4 / (10.4 + F)
        theoretical_p_ratio = 10.4 / (10.4 + current_fm)
        
        is_unsafe_cut = False 
        daily_max_fat_transfer = 0
        
        if phase == 'Bulk':
            consecutive_bulk_days += 1
            weeks_in_bulk = consecutive_bulk_days / 7.0
            fatigue_penalty = weeks_in_bulk * bulk_fatigue_factor
            current_quality = max(0.1, training_quality - fatigue_penalty)
            
            # Apply quality to the p-ratio
            actual_lean_ratio = theoretical_p_ratio * current_quality
            
            delta_weight = surplus / kcal_fat
            lean_gain = delta_weight * actual_lean_ratio
            fat_gain = delta_weight * (1 - actual_lean_ratio)
            current_lm += lean_gain
            current_fm += fat_gain
        else:
            consecutive_bulk_days = 0
            max_fat_energy = current_fm * 69.0
            daily_max_fat_transfer = max_fat_energy
            
            if deficit <= max_fat_energy:
                delta_weight = deficit / kcal_fat
                fat_loss = delta_weight * training_quality
                lean_loss = delta_weight * (1 - training_quality)
            else:
                is_unsafe_cut = True
                fat_loss = max_fat_energy / kcal_fat
                remaining_deficit = deficit - max_fat_energy
                lean_loss_penalty = remaining_deficit / kcal_muscle 
                base_lean_loss = (max_fat_energy / kcal_fat) * (1 - training_quality)
                lean_loss = base_lean_loss + lean_loss_penalty

            current_fm -= fat_loss
            current_lm -= lean_loss

        current_fm = max(0.1, current_fm)
        current_lm = max(20, current_lm)
        current_w = current_lm + current_fm
        current_bf_p = (current_fm / current_w) * 100.0
        
        days_data.append({
            "Week": day_idx / 7.0,
            "Weight": current_w,
            "BodyFat": current_bf_p,
            "LeanMass": current_lm,
            "FatMass": current_fm,
            "Phase": "Unsafe" if is_unsafe_cut else phase,
            "SafeDeficitLimit": daily_max_fat_transfer if phase == "Cut" else 0,
            "PRatio": theoretical_p_ratio # This is the user's p-ratio at this specific point
        })
        
    return pd.DataFrame(days_data)