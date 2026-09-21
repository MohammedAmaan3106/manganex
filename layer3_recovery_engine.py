import json
import os
import pandas as pd
import numpy as np

def run_dynamic_recovery_simulation(
    selected_zone_id="Zone-M001",
    zone_grade_pct=42.0,
    zone_prospectivity=85.0,
    rainfall_mm=42.0,
    equip_avail_pct=78.0,
    blast_delay_hrs=3.0,
    stockpile_available_tons=4000.0,
    target_monthly_tonnage=25000.0
):
    print("=================================================================")
    print(f" MANGANEX: LAYER 3 - CLOSED-LOOP ACTION TWIN FOR [{selected_zone_id}]")
    print("=================================================================")
    
    # Ingest Layer 2 forecast baseline
    l2_path = 'outputs/layer2_production_forecast.json'
    if os.path.exists(l2_path):
        with open(l2_path, 'r') as f:
            l2_data = json.load(f)
            metrics = l2_data.get('forecast_results') or l2_data.get('kpi_metrics', {})
            predicted = float(metrics.get('predicted_production_tonnage', 22850.0))
    else:
        predicted = 22850.0
        
    shortfall = max(0.0, target_monthly_tonnage - predicted)
    
    # 1. Physics-based recovery optimization accounting for clicked zone ore grade
    grade_multiplier = max(0.8, zone_grade_pct / 35.0)
    
    # Scenario A: Reschedule blast sequence away from high moisture windows
    rec_a = round(min(blast_delay_hrs, 4.5) * 250.0, 0)
    
    # Scenario B: Secondary excavator dispatch & haulage route re-routing
    avail_gap = max(0.0, 92.0 - equip_avail_pct)
    rec_b = round(avail_gap * 90.0, 0)
    
    # Scenario C: Selective Face Prioritization on Clicked Zone
    rec_c = round((rec_a * 0.6) + (avail_gap * 60.0 * grade_multiplier) + (zone_prospectivity * 8.5), 0)
    
    # Scenario D: Composite Optimal Action (Fleet + Blast + Selective Face + Strategic Stockpile Blend)
    unresolved = max(0.0, shortfall - (rec_a * 0.8 + rec_b * 0.7 + (zone_prospectivity * 5.0)))
    stockpile_draw = min(stockpile_available_tons, unresolved)
    rec_d = round((rec_a * 0.8) + (rec_b * 0.7) + (zone_prospectivity * 5.0) + stockpile_draw, 0)
    
    scenarios = [
        {
            "scenario_id": "Scenario A",
            "name": "Dynamic Blast Window Rescheduling",
            "recovered_tonnage": rec_a,
            "revised_production": round(predicted + rec_a, 0),
            "shortfall_remaining": round(max(0.0, shortfall - rec_a), 0),
            "status": "Resolved" if (predicted + rec_a) >= target_monthly_tonnage else "Partial Recovery"
        },
        {
            "scenario_id": "Scenario B",
            "name": "Fleet Redeployment & Shift Extension",
            "recovered_tonnage": rec_b,
            "revised_production": round(predicted + rec_b, 0),
            "shortfall_remaining": round(max(0.0, shortfall - rec_b), 0),
            "status": "Resolved" if (predicted + rec_b) >= target_monthly_tonnage else "Partial Recovery"
        },
        {
            "scenario_id": "Scenario C",
            "name": f"Selective Face Prioritization ({selected_zone_id})",
            "recovered_tonnage": rec_c,
            "revised_production": round(predicted + rec_c, 0),
            "shortfall_remaining": round(max(0.0, shortfall - rec_c), 0),
            "status": "Resolved" if (predicted + rec_c) >= target_monthly_tonnage else "Partial Recovery"
        },
        {
            "scenario_id": "Scenario D",
            "name": "Composite Multi-Vector Strategy (Recommended)",
            "recovered_tonnage": rec_d,
            "revised_production": round(predicted + rec_d, 0),
            "shortfall_remaining": round(max(0.0, shortfall - rec_d), 0),
            "remaining_stockpile": round(stockpile_available_tons - stockpile_draw, 0),
            "status": "Shortfall Resolved" if (predicted + rec_d) >= target_monthly_tonnage else "Substantial Recovery"
        }
    ]
    
    # 2. Automated Step-by-Step Prescriptive Mine Manager Directives
    sop_directives = [
        f"1. DISPATCH ORDER: Re-route 2x 50-T Dumpers and 1x Excavator fleet directly to face [{selected_zone_id}] (Estimated Grade: {zone_grade_pct:.1f}% Mn).",
        f"2. BLAST TIMING: Advance the 14:00 blast window by 90 minutes to evade satellite-predicted {rainfall_mm:.0f}mm rainfall infiltration.",
        f"3. BENEFICIATION BLENDING: Blend {stockpile_draw:,.0f} T from Stockpile Yard-3 with fresh feed from {selected_zone_id} to maintain constant 38% refinery grade.",
        f"4. FLEET MAINTENANCE: Schedule rapid 45-minute pit-stop servicing during the blasting evacuation clearance interval."
    ]
    
    output = {
        "selected_zone": {
            "cell_id": selected_zone_id,
            "grade_pct": zone_grade_pct,
            "prospectivity_score": zone_prospectivity
        },
        "baseline": {
            "target_tons": target_monthly_tonnage,
            "predicted_tons": predicted,
            "shortfall_tons": shortfall
        },
        "scenarios": scenarios,
        "recommended_action": scenarios[3],
        "prescriptive_manager_directives": sop_directives
    }
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/layer3_action_twin_recommendations.json', 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"[SUCCESS] Closed-loop simulation complete for {selected_zone_id}.")
    return output

if __name__ == '__main__':
    run_dynamic_recovery_simulation()