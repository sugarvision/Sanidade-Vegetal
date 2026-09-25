import os
import pandas as pd
from pathlib import Path

def test_feature_numerical_limits():
    """
    Test script to validate numerical limits of the extracted features.
    """
    base_dir = Path(__file__).resolve().parent.parent
    output_csv = base_dir / 'data' / 'processed' / 'abt_features_extracted.csv'
    
    assert output_csv.exists(), f"Feature extraction output not found: {output_csv}"
    
    df = pd.read_csv(output_csv)
    
    assert not df.empty, "The extracted features DataFrame is empty."
    
    # Check for expected columns
    expected_cols = ['mean_hue', 'std_saturation', 'exg_index', 'exr_index', 'glcm_contrast', 'glcm_homogeneity']
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Validate numerical limits
    # mean_hue is typically between 0 and 180 in OpenCV (HSV), but here we might generate up to ~180
    assert df['mean_hue'].between(0, 180).all(), "mean_hue out of bounds [0, 180]"
    
    # std_saturation should be > 0 and typically < 1 if normalized
    assert df['std_saturation'].between(0, 1).all(), "std_saturation out of bounds [0, 1]"
    
    # exg_index and exr_index limits (can be negative or positive depending on pixel values)
    assert df['exg_index'].between(-255, 510).all(), "exg_index out of bounds"
    assert df['exr_index'].between(-255, 510).all(), "exr_index out of bounds"
    
    # GLCM homogeneity is between 0 and 1
    assert df['glcm_homogeneity'].between(0, 1).all(), "glcm_homogeneity out of bounds [0, 1]"
    
    # GLCM contrast is >= 0
    assert (df['glcm_contrast'] >= 0).all(), "glcm_contrast must be non-negative"

    print("All numerical limits validated successfully!")

if __name__ == "__main__":
    test_feature_numerical_limits()
