# src/features/feature_extraction.py

import pandas as pd

def extract_features(df):
    """
    Simple feature extraction for physical activity dataset.
    This function assumes numerical columns except 'Activity_Type' are features.
    """
    # If dataset has time or non-numerical columns, drop them
    if 'Activity_Type' in df.columns:
        X = df.drop(columns=['Activity_Type'])
    else:
        X = df.copy()
    
    # Return feature dataframe
    return X
