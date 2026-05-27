"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [10.5, 20.3, 30.1, 40.2, 50.4],
        'category': ['A', 'B', 'A', 'B', 'A'],
        'date': pd.date_range('2024-01-01', periods=5),
    })


@pytest.fixture
def dataframe_with_issues():
    """Create dataframe with data quality issues."""
    return pd.DataFrame({
        'id': [1, 1, 2, 3, 3, 4],
        'value': [10.0, np.nan, 20.0, 999.0, 30.0, np.nan],
        'category': ['A', 'A', 'B', np.nan, 'A', 'C'],
        'date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01'),
                 pd.Timestamp('2024-01-02'), None, 
                 pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-04')],
    })


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_excel_file(sample_dataframe):
    """Create temporary Excel file."""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        temp_path = f.name
    
    sample_dataframe.to_excel(temp_path, index=False)
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_schema():
    """Create mock schema information."""
    return {
        "column_roles": {
            "id": "identifier",
            "value": "numerical",
            "category": "categorical",
            "date": "datetime",
        }
    }


@pytest.fixture
def mock_semantic_schema():
    """Create mock semantic schema."""
    return {
        "id": {
            "type": "numeric",
            "semantic": "identifier",
            "confidence": 0.95,
        },
        "value": {
            "type": "numeric",
            "semantic": "continuous",
            "confidence": 0.90,
        },
        "category": {
            "type": "categorical",
            "semantic": "categorical",
            "confidence": 0.85,
        },
        "date": {
            "type": "datetime",
            "semantic": "timestamp",
            "confidence": 0.95,
        },
    }
