# MANGANEX

Next-generation geospatial intelligence and predictive dispatch engine engineered to discover hidden manganese deposits, forecast volumetric extraction risks, and execute real-time operational re-routing to guarantee production targets.

Developed for **Smart India Hackathon 2026** (Problem Statement: **PS-26009**) by **Team HEXATRON**.

---

## Core Capabilities

* **30-Day Trajectory Forecasting:** Leverages gradient-boosted regression (XGBoost) with 95% confidence intervals to pinpoint extraction deficits weeks before they impact quotas.
* **Game-Theoretic Root-Cause Attribution:** Employs TreeSHAP to decompose complex yield bottlenecks into transparent, percentage-weighted drivers (e.g., rainfall levels, bench slope failure, equipment downtime, blast delays).
* **Target-Driven Re-Routing Engine:** Solves multi-objective constrained optimization problems (via SciPy) to dynamically redistribute excavators, haulage trucks, and extraction priorities across neighboring pits ($\le$ 15 km) to hit target production margins.
* **Explainable Digital Twin:** Pre-computes operational "what-if" re-routing scenarios under DGMS regulatory safety constraints before executing field dispatches.
* **Continuous Event-Driven Orchestration:** Employs n8n pipelines for continuous satellite ingestion, telemetry polling, and automated model retraining.
* **Subsurface Voxel Inversion (-250 m):** Fuses Copernicus Sentinel-1 SAR backscatter with Sentinel-2 multi-spectral bands via the Prithvi-EO foundation model to reconstruct 3D lithological strata down to -250 meters.

---

## The Closed-Loop Cycle

1. **Explore:** Ingests SAR/MSI imagery, geological lithology, and rainfall metrics.
2. **Predict:** Projects 30-day production trends and flags shortfall risks.
3. **Simulate:** Runs digital-twin simulations against operational thresholds and DGMS safety benchmarks.
4. **Analyse:** Decodes operational root causes using TreeSHAP explainability.
5. **Mine & Re-Route:** Dispatches real-time re-routing directives to field crews to neutralize production deficits.

---

## Architecture & Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Forecasting & Optimization** | Python, XGBoost, TreeSHAP, SciPy (Linear Dispatch Optimizer) |
| **Pipeline Automation** | n8n Workflow Automation |
| **Mission Control** | Modular HTML5/CSS3/JavaScript GIS & Telemetry Cockpit |
| **Earth Observation & AI** | Sentinel-1 SAR, Sentinel-2 Multi-Spectral, NASA/IBM Prithvi-EO (Hugging Face) |

---

## Quickstart

### 1. Clone & Set Up Backend
```bash
git clone [https://github.com/MohammedAmaan3106/manganex.git](https://github.com/MohammedAmaan3106/manganex.git)
cd manganex
pip install -r requirements.txt
