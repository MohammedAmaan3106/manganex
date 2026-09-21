import time
from layer1_feature_extractor import generate_comprehensive_indian_manganese_dataset
from layer1_prithvi_foundation import extract_deep_prithvi_representations
from layer1_prospectivity_model import execute_prospectivity_engine

def run():
    print("=================================================================")
    print("  MANGANEX: LAYER 1 - SPACE & ORE INTELLIGENCE ENGINE (BEAST MODE) ")
    print("  IBM-NASA Prithvi-EO 2.0 Foundation Model + Earth Observation   ")
    print("=================================================================")
    start_time = time.time()
    
    print("\n[PHASE 1] Generating Comprehensive Indian Manganese & EO Belts...")
    generate_comprehensive_indian_manganese_dataset()
    
    print("\n[PHASE 2] Extracting Prithvi-EO 2.0 ViT Representations (GPU FP16)...")
    extract_deep_prithvi_representations()
    
    print("\n[PHASE 3] Feature Fusion & Training High-Precision Prospectivity Model...")
    execute_prospectivity_engine()
    
    elapsed = time.time() - start_time
    print(f"\n[COMPLETE] Layer 1 Engine fully executed in {elapsed:.2f} seconds.")
    print("Outputs generated in ./outputs/ folder ready for Layer 2 and Layer 3.")

if __name__ == '__main__':
    run()