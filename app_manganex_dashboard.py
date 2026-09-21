import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import folium
from folium.plugins import HeatMap, Fullscreen
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

from layer2_production_model import train_and_run_production_intelligence
from layer3_recovery_engine import run_dynamic_recovery_simulation
from layer3_autonomous_dispatch import compute_multi_pit_dispatch_optimization

# Streamlit Page Setup
st.set_page_config(
    page_title="MANGANEX | AI Mine Command Center",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Manganese Black & Ore Gold CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #080A0F;
        color: #F8FAFC;
    }

    @keyframes goldGlow {
        0% { border-color: rgba(250, 204, 21, 0.25); box-shadow: 0 0 10px rgba(250, 204, 21, 0.1); }
        50% { border-color: rgba(250, 204, 21, 0.6); box-shadow: 0 0 20px rgba(250, 204, 21, 0.25); }
        100% { border-color: rgba(250, 204, 21, 0.25); box-shadow: 0 0 10px rgba(250, 204, 21, 0.1); }
    }

    .gold-card {
        background: linear-gradient(135deg, #0D121C 0%, #131A26 100%);
        border: 1px solid rgba(250, 204, 21, 0.35);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        animation: goldGlow 4s infinite ease-in-out;
    }

    .gold-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 800;
        color: #FACC15;
        letter-spacing: 1.2px;
        margin-bottom: 2px;
        text-shadow: 0 0 12px rgba(250, 204, 21, 0.35);
    }

    .gold-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #94A3B8;
        letter-spacing: 0.8px;
        margin-bottom: 16px;
    }

    .sop-directive {
        background-color: #05070B;
        border-left: 4px solid #FACC15;
        border-radius: 4px;
        padding: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #FDE047;
        margin-bottom: 8px;
    }

    div[data-testid="stMetricValue"] {
        color: #FACC15 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Data Ingestion
@st.cache_data
def load_datasets():
    df_ranked = pd.read_csv('outputs/manganese_exploration_ranked.csv')
    with open('outputs/layer1_intelligence_summary.json', 'r') as f:
        l1 = json.load(f)
    return df_ranked, l1

df_ranked, l1_summary = load_datasets()

# State Management for Selected Zone
if 'selected_zone' not in st.session_state:
    top_init = df_ranked.iloc[0]
    st.session_state.selected_zone = {
        'cell_id': str(top_init['cell_id']),
        'sector': str(top_init['sector_name']),
        'lat': float(top_init['latitude']),
        'lon': float(top_init['longitude']),
        'prospectivity': float(top_init['prospectivity_score']),
        'confidence': float(top_init['confidence_score']),
        'grade': float(top_init['ore_grade_pct']),
        'rank': str(top_init['exploration_priority_rank']),
        'ndvi': float(top_init['ndvi']),
        'lst': float(top_init['lst_celsius']),
        'soil_moisture': float(top_init['soil_moisture_pct']),
        'elevation': float(top_init['elevation_m']),
        'slope': float(top_init['slope_deg']),
        'iron_oxide': float(top_init['iron_oxide_index']),
        'clay': float(top_init['clay_alteration_index'])
    }

# Sidebar Navigation
st.sidebar.markdown("<h2 style='color: #FACC15; font-family: monospace; letter-spacing: 2px; margin-bottom: 0;'>MANGANEX</h2>", unsafe_allow_html=True)
st.sidebar.caption("MOIL LTD. / MINISTRY OF STEEL")

selected_page = st.sidebar.radio(
    "NAVIGATION (5 DASHBOARDS):",
    [
        "1. Mine Command Center",
        "2. Ore Intelligence (GIS)",
        "3. Production Risk & SHAP",
        "4. Recovery Lab (Action Twin)",
        "5. Autonomous Dispatch Re-Router"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(f"**Active Target:** `{st.session_state.selected_zone['cell_id']}`")
st.sidebar.caption(f"Sector: {st.session_state.selected_zone['sector']}\nGrade: {st.session_state.selected_zone['grade']:.1f}% Mn")

# ==============================================================================
# DASHBOARD 1: MINE COMMAND CENTER
# ==============================================================================
if selected_page == "1. Mine Command Center":
    st.markdown('<p class="gold-header">1. MINE COMMAND CENTER</p>', unsafe_allow_html=True)
    st.markdown('<p class="gold-sub">Pan-India Mineral Logistics Telemetry & Real-Time Production Risk</p>', unsafe_allow_html=True)

    l2_res = train_and_run_production_intelligence(
        target_monthly_tonnage=25000.0,
        active_rainfall_mm=42.0,
        active_equip_availability_pct=78.0,
        active_blast_delay_hrs=3.0,
        active_ore_accessibility_score=float(st.session_state.selected_zone['prospectivity'] * 0.8)
    )
    kpi = l2_res['forecast_results']

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="gold-card">', unsafe_allow_html=True)
        st.caption("TARGET PRODUCTION")
        st.metric("Target", f"{kpi['target_production_tonnage']:,.0f} T", label_visibility="collapsed")
        st.caption("Planned Monthly Quota")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="gold-card">', unsafe_allow_html=True)
        st.caption("AI PRODUCTION FORECAST")
        st.metric("Forecast", f"{kpi['predicted_production_tonnage']:,.0f} T", delta=f"-{kpi['expected_shortfall_tonnage']:,.0f} T", delta_color="inverse", label_visibility="collapsed")
        st.caption("Predicted Shortfall Deficit")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="gold-card">', unsafe_allow_html=True)
        st.caption("SHORTFALL RISK LEVEL")
        st.metric("Risk", kpi['shortfall_risk_level'], label_visibility="collapsed")
        st.caption(f"Weather Risk: {kpi['weather_risk_level']}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="gold-card">', unsafe_allow_html=True)
        st.caption("95% CONFIDENCE INTERVAL")
        st.metric("Bounds", f"{kpi['confidence_interval']['lower_bound_tonnage']:,.0f} - {kpi['confidence_interval']['upper_bound_tonnage']:,.0f} T", label_visibility="collapsed")
        st.caption("Residual Precision")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    
    col_gauge, col_radar = st.columns([1, 1])
    
    with col_gauge:
        st.subheader("Equipment Fleet Availability Gauge")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=78.0,
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#FACC15'},
                'bar': {'color': "#FACC15"},
                'steps': [
                    {'range': [0, 60], 'color': '#1E293B'},
                    {'range': [60, 80], 'color': '#0F172A'},
                    {'range': [80, 100], 'color': '#0284C7'}
                ]
            }
        ))
        fig_g.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#F8FAFC"))
        st.plotly_chart(fig_g, use_container_width=True)

    # RESTORED MULTI-SPECTRAL WEB/RADAR STRUCTURE
    with col_radar:
        st.subheader("Active Pit Multi-Spectral Web Signature (Radar)")
        categories = ['Iron Oxide Alteration', 'Clay Alteration', 'Ferrous Minerals', 'DEM Elevation', 'Thermal LST', 'Sub-Surface Density']
        r_vals = [
            min(100.0, float(st.session_state.selected_zone['iron_oxide']) * 35.0),
            min(100.0, float(st.session_state.selected_zone['clay']) * 40.0),
            82.0,
            min(100.0, (float(st.session_state.selected_zone['elevation']) / 500.0) * 100.0),
            min(100.0, (float(st.session_state.selected_zone['lst']) / 45.0) * 100.0),
            float(st.session_state.selected_zone['prospectivity'])
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=r_vals,
            theta=categories,
            fill='toself',
            line=dict(color='#FACC15', width=2),
            fillcolor='rgba(250, 204, 21, 0.25)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#94A3B8", gridcolor="#1E293B"),
                angularaxis=dict(color="#FACC15", gridcolor="#1E293B")
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            height=260,
            margin=dict(l=30, r=30, t=10, b=10),
            showlegend=False,
            font=dict(color="#F8FAFC", family="monospace")
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ==============================================================================
# DASHBOARD 2: ORE INTELLIGENCE (GIS)
# ==============================================================================
elif selected_page == "2. Ore Intelligence (GIS)":
    st.markdown('<p class="gold-header">2. ORE INTELLIGENCE (NASA / ESRI SATELLITE)</p>', unsafe_allow_html=True)
    st.markdown('<p class="gold-sub">Geospatial Mineral Prospectivity Map & Strategic Drilling Targets</p>', unsafe_allow_html=True)

    col_map, col_details = st.columns([3, 2])
    with col_map:
        st.subheader("Geospatial Mineral Heatmap")
        center_lat = st.session_state.selected_zone['lat']
        center_lon = st.session_state.selected_zone['lon']

        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri World Imagery')
        heat_data = [[r['latitude'], r['longitude'], r['prospectivity_score']] for _, r in df_ranked.iterrows()]
        HeatMap(heat_data, radius=16, blur=14, max_zoom=1, gradient={0.2: '#0284C7', 0.5: '#FACC15', 0.8: '#F97316', 1.0: '#EF4444'}).add_to(m)

        for _, row in df_ranked.head(20).iterrows():
            is_active = row['cell_id'] == st.session_state.selected_zone['cell_id']
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8 if is_active else 5,
                color='#FACC15' if is_active else '#EF4444',
                fill=True,
                fill_color='#FACC15' if is_active else '#EF4444',
                fill_opacity=0.9,
                tooltip=f"{row['cell_id']} ({row['sector_name']}) | Score: {row['prospectivity_score']}"
            ).add_to(m)

        Fullscreen().add_to(m)
        st_folium(m, width=680, height=480)

        selected_from_list = st.selectbox(
            "Select Target Zone to Re-Target System:",
            options=df_ranked['cell_id'].head(20).tolist(),
            index=df_ranked['cell_id'].head(20).tolist().index(st.session_state.selected_zone['cell_id'])
        )

        if selected_from_list != st.session_state.selected_zone['cell_id']:
            row_match = df_ranked[df_ranked['cell_id'] == selected_from_list].iloc[0]
            st.session_state.selected_zone = {
                'cell_id': str(row_match['cell_id']),
                'sector': str(row_match['sector_name']),
                'lat': float(row_match['latitude']),
                'lon': float(row_match['longitude']),
                'prospectivity': float(row_match['prospectivity_score']),
                'confidence': float(row_match['confidence_score']),
                'grade': float(row_match['ore_grade_pct']),
                'rank': str(row_match['exploration_priority_rank']),
                'ndvi': float(row_match['ndvi']),
                'lst': float(row_match['lst_celsius']),
                'soil_moisture': float(row_match['soil_moisture_pct']),
                'elevation': float(row_match['elevation_m']),
                'slope': float(row_match['slope_deg']),
                'iron_oxide': float(row_match['iron_oxide_index']),
                'clay': float(row_match['clay_alteration_index'])
            }
            st.rerun()

    with col_details:
        st.markdown('<div class="gold-card">', unsafe_allow_html=True)
        st.subheader(f"Zone Telemetry: {st.session_state.selected_zone['cell_id']}")
        st.caption(f"Sector: {st.session_state.selected_zone['sector']}")

        c1, c2 = st.columns(2)
        c1.metric("Prospectivity Score", f"{st.session_state.selected_zone['prospectivity']:.0f} / 100")
        c2.metric("AI Confidence Level", f"{st.session_state.selected_zone['confidence']:.0f} %")

        c3, c4 = st.columns(2)
        c3.metric("Exploration Priority", st.session_state.selected_zone['rank'])
        c4.metric("Assayed Ore Grade", f"{st.session_state.selected_zone['grade']:.1f} % Mn")

        st.markdown("---")
        st.markdown("#### Surface Spectral Indicators")
        st.write(f"• **Iron Oxide Index (B4/B2):** `{st.session_state.selected_zone['iron_oxide']:.3f}`")
        st.write(f"• **Clay Alteration (B11/B12):** `{st.session_state.selected_zone['clay']:.3f}`")
        st.write(f"• **Land Surface Temp:** `{st.session_state.selected_zone['lst']:.1f} °C`")
        st.write(f"• **Pit Bench Slope:** `{st.session_state.selected_zone['slope']:.1f}°`")
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# DASHBOARD 3: PRODUCTION RISK & SHAP (FIXED GRAPH & CLEAR EXPLANATION)
# ==============================================================================
elif selected_page == "3. Production Risk & SHAP":
    st.markdown('<p class="gold-header">3. PRODUCTION RISK & SHAP ROOT CAUSE</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="gold-sub">Forecasting 30-Day Extraction Degradation Curve for Face: <b>{st.session_state.selected_zone["cell_id"]}</b></p>', unsafe_allow_html=True)

    l2_res = train_and_run_production_intelligence(
        target_monthly_tonnage=25000.0,
        active_rainfall_mm=42.0,
        active_equip_availability_pct=78.0,
        active_blast_delay_hrs=3.0,
        active_ore_accessibility_score=float(st.session_state.selected_zone['prospectivity'] * 0.8)
    )

    col_plot, col_shap = st.columns([3, 2])
    
    with col_plot:
        st.subheader("📈 30-Day Extraction Degradation vs Planned Quota")
        
        # Build 30-Day Daily/Weekly Trajectory
        traj = l2_res['forecast_trajectory_30_day']
        df_traj = pd.DataFrame(traj)
        
        # Generate rich Plotly visualization with Area Gap and Confidence Bounds
        fig = go.Figure()
        
        # 1. Target Line (Green Dashed)
        fig.add_trace(go.Scatter(
            x=df_traj['week'],
            y=df_traj['target_tons'],
            mode='lines+markers+text',
            name="Target Quota (6,250 T/wk)",
            line=dict(color='#10B981', width=3, dash='dash'),
            text=[f"{v:,.0f} T" for v in df_traj['target_tons']],
            textposition="top right",
            textfont=dict(color='#10B981', size=10)
        ))
        
        # 2. AI Predicted Trajectory (Gold Solid)
        fig.add_trace(go.Scatter(
            x=df_traj['week'],
            y=df_traj['predicted_tons'],
            mode='lines+markers+text',
            name="AI Forecast Trajectory",
            line=dict(color='#FACC15', width=4),
            text=[f"{v:,.0f} T" for v in df_traj['predicted_tons']],
            textposition="bottom left",
            textfont=dict(color='#FACC15', size=10)
        ))
        
        # 3. Highlighted Deficit Fill Area
        fig.add_trace(go.Scatter(
            x=list(df_traj['week']) + list(df_traj['week'])[::-1],
            y=list(df_traj['target_tons']) + list(df_traj['predicted_tons'])[::-1],
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name="Production Shortfall Gap (Deficit)",
            hoverinfo="skip"
        ))
        
        fig.update_layout(
            height=380,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title="Weekly Output (Tons)", gridcolor="#1E293B", range=[4500, 7000]),
            xaxis=dict(gridcolor="#1E293B"),
            font=dict(color="#F8FAFC", family="monospace"),
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Why this graph matters:** It reveals the progressive extraction deficit starting from Week 2 due to cumulative haul-road mud buildup and blast delays, allowing managers to intervene 14 days before end-of-month penalties.")

    with col_shap:
        st.subheader("🔍 TreeSHAP Root Cause Attribution")
        shap_data = l2_res['explainable_ai_shap_drivers']
        df_s = pd.DataFrame({'Bottleneck Factor': list(shap_data.keys()), 'Impact (%)': list(shap_data.values())}).sort_values(by='Impact (%)', ascending=True)

        fig_s = px.bar(
            df_s,
            x='Impact (%)',
            y='Bottleneck Factor',
            orientation='h',
            color='Impact (%)',
            color_continuous_scale=['#1E293B', '#FACC15', '#EF4444']
        )
        fig_s.update_layout(
            height=380,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#F8FAFC", family="monospace"),
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_s, use_container_width=True)

# ==============================================================================
# DASHBOARD 4: RECOVERY LAB (ACTION TWIN)
# ==============================================================================
elif selected_page == "4. Recovery Lab (Action Twin)":
    st.markdown('<p class="gold-header">4. RECOVERY LAB (WHAT-IF ACTION TWIN)</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="gold-sub">Closed-Loop Optimization for Face: <b>{st.session_state.selected_zone["cell_id"]}</b></p>', unsafe_allow_html=True)

    c_sliders, c_results = st.columns([1, 2])
    with c_sliders:
        st.markdown('<div class="gold-card">', unsafe_allow_html=True)
        st.subheader("Simulation Controls")
        r_rain = st.slider("Rainfall Infiltration (mm)", 0.0, 120.0, 42.0, 2.0)
        r_avail = st.slider("Fleet Availability (%)", 40.0, 100.0, 78.0, 1.0)
        r_blast = st.slider("Blasting Delay (hrs)", 0.0, 12.0, 3.0, 0.5)
        r_stock = st.slider("Stockpile Draw Capacity (T)", 0.0, 10000.0, 4000.0, 250.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_results:
        sim_data = run_dynamic_recovery_simulation(
            selected_zone_id=st.session_state.selected_zone['cell_id'],
            zone_grade_pct=st.session_state.selected_zone['grade'],
            zone_prospectivity=st.session_state.selected_zone['prospectivity'],
            rainfall_mm=r_rain,
            equip_avail_pct=r_avail,
            blast_delay_hrs=r_blast,
            stockpile_available_tons=r_stock
        )

        rec = sim_data['recommended_action']
        st.markdown('<div class="gold-card">', unsafe_allow_html=True)
        st.subheader("Optimized Production Recovery")

        m1, m2 = st.columns(2)
        m1.metric("Revised Production Output", f"{rec['revised_production']:,.0f} T", delta=f"+{rec['recovered_tonnage']:,.0f} T Recovered")
        m2.metric("Optimal Strategy", rec['scenario_id'], delta=rec['name'])

        st.markdown("---")
        st.markdown("#### Prescriptive Mine Manager Directives (SOP)")
        for direct in sim_data['prescriptive_manager_directives']:
            st.markdown(f'<div class="sop-directive">{direct}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# DASHBOARD 5: AUTONOMOUS DISPATCH RE-ROUTER
# ==============================================================================
elif selected_page == "5. Autonomous Dispatch Re-Router":
    st.markdown('<p class="gold-header">5. AUTONOMOUS PIT DISPATCH & SPATIAL RE-ROUTER</p>', unsafe_allow_html=True)
    st.markdown('<p class="gold-sub">Autonomous Goal-Seeking Engine: Scans surrounding Indian mining zones to re-route extraction and prevent bottlenecks.</p>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown('<div class="gold-card">', unsafe_allow_html=True)
        st.subheader("Target Quota Inputs")
        t_quota = st.number_input("Target Quota (Tons):", min_value=5000.0, max_value=60000.0, value=25000.0, step=1000.0)
        s_hours = st.slider("Active Shift Length (Hours):", 6.0, 16.0, 8.0, 1.0)
        f_units = st.slider("Active Fleet Units:", 4, 30, 12, 1)
        r_rain_d = st.slider("Pit Precipitation (mm):", 0.0, 120.0, 42.0, 2.0)
        f_uptime = st.slider("Fleet Uptime (%):", 40.0, 100.0, 78.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        dispatch_data = compute_multi_pit_dispatch_optimization(
            selected_zone_id=st.session_state.selected_zone['cell_id'],
            target_production_tons=t_quota,
            shift_hours=s_hours,
            active_fleet_count=f_units,
            rainfall_mm=r_rain_d,
            fleet_avail_pct=f_uptime
        )

        feas = dispatch_data['feasibility_assessment']
        plan = dispatch_data['multi_pit_rebalanced_plan']

        st.markdown('<div class="gold-card">', unsafe_allow_html=True)
        st.subheader("Feasibility Diagnostic & Multi-Pit Rebalancing")
        d1, d2 = st.columns(2)
        d1.metric("Primary Safe Limit", f"{feas['primary_safe_capacity_tons']:,.0f} T", delta=f"Face: {feas['primary_zone_id']}")
        d2.metric("Guaranteed Delivery", f"{plan['total_delivery_tonnage']:,.0f} T", delta="100% Target Met" if plan['is_target_fully_met'] else "Deficit")

        st.markdown("---")
        st.markdown("#### Executive Spatial Advisory")
        st.markdown(f'<div class="sop-directive">{dispatch_data["executive_advisory"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if dispatch_data['optimized_diversion_routes']:
            st.subheader("🚜 Autonomous Multi-Pit Haulage Allocation Schedule")
            df_routes = pd.DataFrame(dispatch_data['optimized_diversion_routes'])
            st.dataframe(
                df_routes[['route_rank', 'zone_id', 'sector_name', 'allocated_tonnage', 'ore_grade_pct', 'haulage_distance_km', 'recommended_fleet_dispatch', 'terrain_status']],
                hide_index=True,
                use_container_width=True
            )