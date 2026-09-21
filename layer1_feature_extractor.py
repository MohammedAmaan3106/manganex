import os
import numpy as np
import pandas as pd

def build_space_geology_features(num_samples=3000, random_state=42):
    np.random.seed(random_state)
    os.makedirs('data/processed', exist_ok=True)
    
    # Real-world Indian Manganese Ore Mining Sectors
    sectors = [
        {"name": "Balaghat-Bharweli Sector (MP)", "lat": (21.80, 22.00), "lon": (80.15, 80.40), "richness": 0.90},
        {"name": "Dongri Buzurg-Bhandara Sector (MH)", "lat": (21.45, 21.65), "lon": (79.60, 79.90), "richness": 0.85},
        {"name": "Mansar-Nagpur Sector (MH)", "lat": (21.35, 21.50), "lon": (79.20, 79.45), "richness": 0.80},
        {"name": "Chhindwara-Sausar Sector (MP)", "lat": (21.55, 21.75), "lon": (78.85, 79.15), "richness": 0.75},
        {"name": "Keonjhar Sector (Odisha)", "lat": (21.80, 22.10), "lon": (85.25, 85.55), "richness": 0.78},
        {"name": "Barren Background Territory", "lat": (20.50, 21.10), "lon": (78.00, 78.80), "richness": 0.08}
    ]
    
    rows_per_sector = num_samples // len(sectors)
    records = []
    zone_idx = 1
    
    for sector in sectors:
        lat = np.random.uniform(sector["lat"][0], sector["lat"][1], rows_per_sector)
        lon = np.random.uniform(sector["lon"][0], sector["lon"][1], rows_per_sector)
        
        # Structural lineament / fault proximity calculation
        fault_center_lat = np.mean(sector["lat"])
        fault_center_lon = np.mean(sector["lon"])
        fault_dist_km = np.sqrt((lat - fault_center_lat)**2 + (lon - fault_center_lon)**2) * 111.0
        fault_factor = np.exp(-fault_dist_km / 10.0) * sector["richness"]
        
        # Simulated Sentinel-2 Multi-Spectral Reflectance Bands [B2, B3, B4, B8, B11, B12]
        b2_blue = np.random.uniform(0.04, 0.09, rows_per_sector)
        b3_green = np.random.uniform(0.06, 0.12, rows_per_sector)
        b4_red = np.random.uniform(0.05, 0.16, rows_per_sector) + (0.05 * fault_factor)
        b8_nir = np.random.uniform(0.18, 0.42, rows_per_sector) - (0.08 * fault_factor)
        b11_swir1 = np.random.uniform(0.15, 0.38, rows_per_sector) + (0.10 * fault_factor)
        b12_swir2 = np.random.uniform(0.10, 0.32, rows_per_sector) + (0.09 * fault_factor)
        
        # Spectral Alteration Indices
        ndvi = (b8_nir - b4_red) / (b8_nir + b4_red + 1e-6)
        iron_oxide_index = b4_red / (b2_blue + 1e-6)
        clay_alteration_index = b11_swir1 / (b12_swir2 + 1e-6)
        ferrous_index = b12_swir2 / (b8_nir + 1e-6)
        
        # Land Surface Temperature & Soil Moisture
        lst_celsius = np.random.normal(36.0, 3.5, rows_per_sector) + (4.0 * fault_factor)
        soil_moisture_pct = np.clip(np.random.normal(22.0, 4.0, rows_per_sector) - (5.0 * fault_factor), 5.0, 45.0)
        rainfall_mm = np.random.normal(1120, 80, rows_per_sector)
        
        # Subsurface & Topography (DEM)
        elevation_m = np.random.normal(380, 40, rows_per_sector) + (35.0 * fault_factor)
        slope_deg = np.clip(np.random.normal(12.0, 5.0, rows_per_sector) + (4.0 * fault_factor), 1.0, 45.0)
        
        # Drill / Assay Ground Truth
        latent_ore_density = (
            0.28 * iron_oxide_index +
            0.24 * clay_alteration_index +
            0.22 * ferrous_index +
            0.18 * fault_factor -
            0.10 * ndvi +
            np.random.normal(0, 0.10, rows_per_sector)
        )
        
        is_reserve = (latent_ore_density > np.percentile(latent_ore_density, 75)).astype(int)
        ore_grade_pct = np.clip(18.0 + (latent_ore_density * 6.5 * sector["richness"]), 10.0, 52.0)
        
        for i in range(rows_per_sector):
            records.append({
                'cell_id': f"Zone-M{zone_idx:03d}",
                'sector_name': sector["name"],
                'latitude': lat[i],
                'longitude': lon[i],
                'band_blue': b2_blue[i],
                'band_green': b3_green[i],
                'band_red': b4_red[i],
                'band_nir': b8_nir[i],
                'band_swir1': b11_swir1[i],
                'band_swir2': b12_swir2[i],
                'ndvi': ndvi[i],
                'iron_oxide_index': iron_oxide_index[i],
                'clay_alteration_index': clay_alteration_index[i],
                'ferrous_mineral_ratio': ferrous_index[i],
                'lst_celsius': lst_celsius[i],
                'soil_moisture_pct': soil_moisture_pct[i],
                'rainfall_mm': rainfall_mm[i],
                'elevation_m': elevation_m[i],
                'slope_deg': slope_deg[i],
                'fault_distance_km': fault_dist_km[i],
                'ore_grade_pct': ore_grade_pct[i],
                'is_manganese_reserve': is_reserve[i]
            })
            zone_idx += 1
            
    df = pd.DataFrame(records)
    df.to_csv('data/processed/space_geology_features.csv', index=False)
    print(f"[DATA READY] Extracted {len(df)} spatial geological cells across Indian mining corridors.")

if __name__ == '__main__':
    build_space_geology_features()