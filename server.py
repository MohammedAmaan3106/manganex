import os
import json
import sqlite3
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from layer2_production_model import train_and_run_production_intelligence
from layer3_recovery_engine import run_dynamic_recovery_simulation
from layer3_autonomous_dispatch import compute_multi_pit_dispatch_optimization

app = FastAPI(title="Manganex Tactical Core API")

# Enable Cross-Origin Requests for seamless local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "manganex.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def auto_init_db():
    """Initializes tables and seeds data if manganex.db does not exist yet."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Monthly Ledger Overview Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS monthly_production_ledger (
        zone_id TEXT PRIMARY KEY,
        sector_name TEXT,
        month_year TEXT,
        planned_target_tons REAL,
        actual_mined_tons REAL,
        lag_lead_balance_tons REAL,
        target_achieved_pct REAL,
        status TEXT
    )
    """)

    # 2. Weekly Operational Data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS weekly_extraction_telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        zone_id TEXT,
        week_label TEXT,
        planned_quota_tons REAL,
        actual_mined_tons REAL,
        ai_predicted_tons REAL,
        rainfall_mm REAL,
        equipment_uptime_pct REAL,
        blasting_delay_hrs REAL,
        FOREIGN KEY (zone_id) REFERENCES monthly_production_ledger (zone_id)
    )
    """)

    # 3. High-Density 108-Point Continuous Trajectory Table (Actual vs Predicted Curve)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS shift_continuous_telemetry (
        shift_index INTEGER,
        zone_id TEXT,
        actual_value REAL,
        predicted_value REAL,
        PRIMARY KEY (shift_index, zone_id)
    )
    """)

    # Seed Monthly Ledger if empty
    cursor.execute("SELECT COUNT(*) FROM monthly_production_ledger")
    if cursor.fetchone()[0] == 0:
        ledger_records = [
            ("Zone-M100", "Balaghat Belt (MP)", "August 2026", 25000.0, 22850.0, -2150.0, 91.4, "LAGGING"),
            ("Zone-M521", "Bhandara Corridor (MH)", "August 2026", 28000.0, 29120.0, 1120.0, 104.0, "LEADING"),
            ("Zone-M304", "Dongri Buzurg (MH)", "August 2026", 18000.0, 16900.0, -1100.0, 93.8, "LAGGING"),
            ("Zone-M812", "Chikla Pit (MH)", "August 2026", 15000.0, 15300.0, 300.0, 102.0, "LEADING")
        ]

        weekly_records = [
            ("Zone-M100", "Wk 1", 6250.0, 6100.0, 6080.0, 18.0, 88.0, 1.5),
            ("Zone-M100", "Wk 2", 6250.0, 5800.0, 5750.0, 42.0, 78.0, 3.0),
            ("Zone-M100", "Wk 3", 6250.0, 5500.0, 5520.0, 65.0, 72.0, 4.2),
            ("Zone-M100", "Wk 4", 6250.0, 5450.0, 5500.0, 58.0, 70.0, 3.8),
            ("Zone-M521", "Wk 1", 7000.0, 7200.0, 7150.0, 8.0, 94.0, 0.8),
            ("Zone-M521", "Wk 2", 7000.0, 7350.0, 7300.0, 12.0, 92.0, 1.0),
            ("Zone-M521", "Wk 3", 7000.0, 7280.0, 7250.0, 15.0, 90.0, 1.2),
            ("Zone-M521", "Wk 4", 7000.0, 7290.0, 7320.0, 10.0, 95.0, 0.5)
        ]

        cursor.executemany("INSERT OR REPLACE INTO monthly_production_ledger VALUES (?,?,?,?,?,?,?,?)", ledger_records)
        cursor.executemany("INSERT OR REPLACE INTO weekly_extraction_telemetry (zone_id, week_label, planned_quota_tons, actual_mined_tons, ai_predicted_tons, rainfall_mm, equipment_uptime_pct, blasting_delay_hrs) VALUES (?,?,?,?,?,?,?,?)", weekly_records)

    # Seed 108 Continuous Shift Telemetry Points if empty
    cursor.execute("SELECT COUNT(*) FROM shift_continuous_telemetry")
    if cursor.fetchone()[0] == 0:
        np.random.seed(42)
        shift_count = 108
        x = np.arange(shift_count)
        
        # Generates the multi-cyclical upward trending trajectory matching the reference graph
        base_trend = 15.0 + 0.16 * x + 4.0 * np.sin(x / 6.0) - 2.5 * np.cos(x / 3.2)
        noise = np.random.normal(0, 0.85, size=shift_count)
        actual_curve = np.clip(base_trend + noise, 11.0, 35.5)

        # Predicted curve tracks actual curve with slight XGBoost residual variance
        pred_noise = np.random.normal(0, 0.5, size=shift_count)
        pred_curve = actual_curve * 0.985 + pred_noise

        shift_rows = []
        for zone in ["Zone-M100", "Zone-M521", "Zone-M304", "Zone-M812"]:
            # Minor variance offsets per zone
            offset = 1.05 if "521" in zone else (0.92 if "304" in zone else 1.0)
            for i in range(shift_count):
                shift_rows.append((
                    int(i),
                    zone,
                    round(float(actual_curve[i] * offset), 2),
                    round(float(pred_curve[i] * offset), 2)
                ))

        cursor.executemany("INSERT OR REPLACE INTO shift_continuous_telemetry VALUES (?,?,?,?)", shift_rows)

    conn.commit()
    conn.close()

auto_init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")

class SimulationRequest(BaseModel):
    selected_zone_id: str = "Zone-M100"
    rainfall_mm: float = 42.0
    fleet_avail_pct: float = 78.0
    blast_delay_hrs: float = 3.0
    stockpile_tons: float = 4000.0
    target_tons: float = 25000.0

@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")

@app.get("/api/initial-telemetry")
def get_initial_telemetry():
    df_ranked = pd.read_csv('outputs/manganese_exploration_ranked.csv')
    with open('outputs/layer1_intelligence_summary.json', 'r') as f:
        l1 = json.load(f)
    with open('outputs/layer2_production_forecast.json', 'r') as f:
        l2 = json.load(f)
        
    return {
        "ranked_zones": df_ranked.to_dict(orient="records"),
        "layer1_summary": l1,
        "layer2_baseline": l2
    }

@app.get("/api/db/history/{zone_id}")
def get_zone_history(zone_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM monthly_production_ledger WHERE zone_id = ?", (zone_id,))
    ledger_row = cursor.fetchone()
    
    cursor.execute("SELECT * FROM weekly_extraction_telemetry WHERE zone_id = ? ORDER BY id ASC", (zone_id,))
    weekly_rows = cursor.fetchall()
    
    if not weekly_rows:
        cursor.execute("SELECT * FROM weekly_extraction_telemetry WHERE zone_id = 'Zone-M100' ORDER BY id ASC")
        weekly_rows = cursor.fetchall()

    conn.close()
    
    return {
        "monthly_ledger": dict(ledger_row) if ledger_row else None,
        "weekly_records": [dict(r) for r in weekly_rows]
    }

@app.get("/api/db/continuous-shifts/{zone_id}")
def get_continuous_shifts(zone_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT shift_index, actual_value, predicted_value FROM shift_continuous_telemetry WHERE zone_id = ? ORDER BY shift_index ASC",
        (zone_id,)
    )
    rows = cursor.fetchall()
    
    if not rows:
        cursor.execute(
            "SELECT shift_index, actual_value, predicted_value FROM shift_continuous_telemetry WHERE zone_id = 'Zone-M100' ORDER BY shift_index ASC"
        )
        rows = cursor.fetchall()
        
    conn.close()
    return {"shifts": [dict(r) for r in rows]}

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    df_ranked = pd.read_csv('outputs/manganese_exploration_ranked.csv')
    matched = df_ranked[df_ranked['cell_id'] == req.selected_zone_id]
    if matched.empty:
        matched = df_ranked.iloc[0]
    else:
        matched = matched.iloc[0]
        
    l2_res = train_and_run_production_intelligence(
        target_monthly_tonnage=req.target_tons,
        active_rainfall_mm=req.rainfall_mm,
        active_equip_availability_pct=req.fleet_avail_pct,
        active_blast_delay_hrs=req.blast_delay_hrs,
        active_ore_accessibility_score=float(matched['prospectivity_score'] * 0.8)
    )
    
    l3_res = run_dynamic_recovery_simulation(
        selected_zone_id=str(matched['cell_id']),
        zone_grade_pct=float(matched['ore_grade_pct']),
        zone_prospectivity=float(matched['prospectivity_score']),
        rainfall_mm=req.rainfall_mm,
        equip_avail_pct=req.fleet_avail_pct,
        blast_delay_hrs=req.blast_delay_hrs,
        stockpile_available_tons=req.stockpile_tons,
        target_monthly_tonnage=req.target_tons
    )
    
    dispatch_res = compute_multi_pit_dispatch_optimization(
        selected_zone_id=str(matched['cell_id']),
        target_production_tons=req.target_tons,
        rainfall_mm=req.rainfall_mm,
        fleet_avail_pct=req.fleet_avail_pct,
        blast_delay_hrs=req.blast_delay_hrs
    )
    
    return {
        "zone_telemetry": matched.to_dict(),
        "layer2": l2_res,
        "layer3_recovery": l3_res,
        "dispatch_plan": dispatch_res
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

   