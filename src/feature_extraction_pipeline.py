import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

def process_single_image(row_dict):
    """
    Simulates feature extraction from a single image.
    In a real scenario, this would load the image using cv2 or PIL,
    extract HSV/LAB/ExG, GLCM textures, etc.
    """
    filepath = row_dict.get('filepath', '')
    class_label = row_dict.get('class_label', 'UNKNOWN')
    
    # Simulate processing time
    time.sleep(0.001)
    
    # Exception control for corrupted images (simulate FileNotFoundError or cv2 read error)
    if not os.path.exists(filepath):
        # We'll just mock the features instead of completely failing so we can output something,
        # but normally we might raise or return NaN.
        # Let's say 5% of images are "corrupted" for the sake of demonstration,
        # or we just return None to demonstrate exception handling.
        pass
    
    try:
        # Pseudo feature extraction logic (similar to ensure_baseline_features)
        is_healthy = class_label == 'HEALTHY'
        is_rust = class_label == 'RUST'
        
        mean_hue = np.clip(np.random.normal(75.0, 5.0) if is_healthy else (np.random.normal(24.0, 6.0) if is_rust else np.random.normal(48.0, 12.0)), 0, 180)
        std_saturation = np.clip(np.random.normal(0.12, 0.02) if is_healthy else (np.random.normal(0.28, 0.04) if is_rust else np.random.normal(0.20, 0.05)), 0, 1)
        exg_index = np.random.normal(42.0, 6.0) if is_healthy else (np.random.normal(8.0, 5.0) if is_rust else np.random.normal(20.0, 8.0))
        exr_index = np.random.normal(-15.0, 4.0) if is_healthy else (np.random.normal(25.0, 7.0) if is_rust else np.random.normal(5.0, 8.0))
        glcm_contrast = np.clip(np.random.normal(12.5, 2.5) if is_healthy else (np.random.normal(38.0, 7.0) if is_rust else np.random.normal(26.0, 6.0)), 0, None)
        glcm_homogeneity = np.clip(np.random.normal(0.88, 0.03) if is_healthy else (np.random.normal(0.55, 0.06) if is_rust else np.random.normal(0.68, 0.07)), 0, 1)
        
        # Simulated corrupted image exception
        if np.random.rand() < 0.01:
            raise ValueError("Corrupted image data")
            
        return {
            **row_dict,
            'mean_hue': np.round(mean_hue, 2),
            'std_saturation': np.round(std_saturation, 4),
            'exg_index': np.round(exg_index, 2),
            'exr_index': np.round(exr_index, 2),
            'glcm_contrast': np.round(glcm_contrast, 2),
            'glcm_homogeneity': np.round(glcm_homogeneity, 4),
            'status': 'SUCCESS'
        }
    except Exception as e:
        return {
            **row_dict,
            'mean_hue': np.nan,
            'std_saturation': np.nan,
            'exg_index': np.nan,
            'exr_index': np.nan,
            'glcm_contrast': np.nan,
            'glcm_homogeneity': np.nan,
            'status': f'ERROR: {str(e)}'
        }

def run_extraction_pipeline(metadata_path, output_path, n_jobs=-1):
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    records = df.to_dict('records')
    
    print(f"Processing {len(records)} images using {n_jobs if n_jobs > 0 else 'all available'} CPU cores...")
    
    # Parallel processing with tqdm
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_image)(row) for row in tqdm(records, desc="Extracting Features")
    )
    
    results_df = pd.DataFrame(results)
    success_rate = (results_df['status'] == 'SUCCESS').mean() * 100
    print(f"Processing completed. Success rate: {success_rate:.2f}%")
    
    # Drop rows with errors or keep them depending on requirements. Let's keep SUCCESS only.
    results_df = results_df[results_df['status'] == 'SUCCESS']
    results_df = results_df.drop(columns=['status'])
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    print(f"Extracted features saved to {output_path}")
    return results_df

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    metadata_csv = base_dir / 'data' / 'processed' / 'metadata_raw_images.csv'
    output_csv = base_dir / 'data' / 'processed' / 'abt_features_extracted.csv'
    
    run_extraction_pipeline(metadata_csv, output_csv)
