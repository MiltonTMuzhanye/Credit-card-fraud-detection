import pytest
import pandas as pd
import numpy as np
from src.features.build_features import FeatureEngineer

def test_create_time_features():
    """Test time feature creation"""
    df = pd.DataFrame({
        'Time': [0, 3600, 7200, 10800],
        'Amount': [100, 200, 300, 400]
    })
    
    engineer = FeatureEngineer()
    df_processed = engineer.create_time_features(df)
    
    assert 'Hour' in df_processed.columns
    assert 'Time_of_Day' in df_processed.columns
    assert df_processed['Hour'].iloc[1] == 1

def test_preprocessor():
    """Test preprocessor creation"""
    engineer = FeatureEngineer()
    preprocessor = engineer.build_preprocessor()
    
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2