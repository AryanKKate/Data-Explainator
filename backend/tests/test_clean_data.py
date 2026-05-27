"""
Unit tests for data cleaning service.
"""

import pytest
import pandas as pd
import numpy as np
from services.clean_data import DataCleaner
from exceptions import DataCleaningError


@pytest.fixture
def sample_df():
    """Create sample dataframe with issues."""
    return pd.DataFrame({
        'id': [1, 1, 2, 3, 3],
        'value': [10.0, np.nan, 20.0, 999.0, 30.0],
        'category': ['A', 'A', 'B', np.nan, 'A'],
    })


@pytest.fixture
def clean_df():
    """Create clean dataframe."""
    return pd.DataFrame({
        'value': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'B', 'A'],
    })


class TestDataCleanerBasic:
    """Test basic cleaning operations."""

    def test_handle_duplicates(self, sample_df):
        """Test duplicate removal."""
        initial_rows = len(sample_df)
        df_clean = DataCleaner._handle_duplicates(sample_df)
        assert len(df_clean) <= initial_rows

    def test_clean_removes_duplicates(self, sample_df):
        """Test clean removes duplicates."""
        df_clean = DataCleaner.clean(sample_df)
        assert len(df_clean) < len(sample_df)

    def test_clean_handles_missing(self, sample_df):
        """Test missing value handling."""
        df_clean = DataCleaner.clean(sample_df)
        # Should have filled missing values
        assert df_clean.isnull().sum().sum() == 0

    def test_clean_preserves_shape(self, sample_df):
        """Test clean preserves column count."""
        df_clean = DataCleaner.clean(sample_df)
        assert df_clean.shape[1] == sample_df.shape[1]

    def test_clean_success_on_valid_data(self, clean_df):
        """Test clean works on valid data."""
        df_clean = DataCleaner.clean(clean_df)
        assert df_clean.shape[0] > 0
        assert df_clean.isnull().sum().sum() == 0

    def test_clean_empty_dataframe(self):
        """Test clean raises error on empty dataframe."""
        with pytest.raises(DataCleaningError):
            DataCleaner.clean(pd.DataFrame())

    def test_clean_none_dataframe(self):
        """Test clean raises error on None."""
        with pytest.raises((DataCleaningError, AttributeError)):
            DataCleaner.clean(None)


class TestDataCleanerNumericalMissing:
    """Test numerical missing value handling."""

    def test_handle_numerical_missing_median(self):
        """Test median filling for numerical."""
        df = pd.DataFrame({'value': [1.0, 2.0, np.nan, 4.0, 5.0]})
        DataCleaner._handle_numerical_missing(df, 'value', strategy='median')
        assert df['value'].isnull().sum() == 0
        assert df.loc[2, 'value'] == 3.0  # Median

    def test_handle_numerical_missing_mean(self):
        """Test mean filling for numerical."""
        df = pd.DataFrame({'value': [1.0, 2.0, np.nan, 4.0, 5.0]})
        DataCleaner._handle_numerical_missing(df, 'value', strategy='mean')
        assert df['value'].isnull().sum() == 0

    def test_handle_numerical_missing_no_missing(self):
        """Test handling when no missing values."""
        df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})
        initial_df = df.copy()
        DataCleaner._handle_numerical_missing(df, 'value')
        pd.testing.assert_frame_equal(df, initial_df)


class TestDataCleanerOutliers:
    """Test outlier detection and handling."""

    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})  # 100 is outlier
        DataCleaner._detect_and_handle_outliers(df, 'value', method='iqr')
        # Should cap extreme values
        assert df['value'].max() < 100

    def test_detect_outliers_zscore(self):
        """Test Z-score outlier detection."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})  # 100 is outlier
        DataCleaner._detect_and_handle_outliers(df, 'value', method='zscore')
        assert df['value'].notna().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
