"""
Unit tests for data loading service.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from services.load_data import DataLoader
from exceptions import (
    FileOperationError,
    FileNotSupportedError,
    FileSizeExceededError,
    DataValidationError,
)


@pytest.fixture
def sample_csv():
    """Create temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,value\n1,10\n2,20\n3,30\n")
        return f.name


@pytest.fixture
def empty_csv():
    """Create empty CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("")
        return f.name


class TestDataLoaderFileValidation:
    """Test file validation logic."""

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileOperationError):
            DataLoader.load("/nonexistent/file.csv")

    def test_load_unsupported_format(self):
        """Test loading unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"test data")
            f.flush()
            
            with pytest.raises(FileNotSupportedError):
                DataLoader.load(f.name)
            
            os.unlink(f.name)

    def test_load_csv_success(self, sample_csv):
        """Test successful CSV loading."""
        try:
            df = DataLoader.load(sample_csv)
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (3, 2)
            assert list(df.columns) == ['id', 'value']
        finally:
            os.unlink(sample_csv)

    def test_load_empty_file(self, empty_csv):
        """Test loading empty file raises error."""
        try:
            with pytest.raises(DataValidationError):
                DataLoader.load(empty_csv)
        finally:
            os.unlink(empty_csv)


class TestDataLoaderFileSize:
    """Test file size validation."""

    def test_validate_file_size(self):
        """Test file size validation works."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,value\n" + "1,10\n" * 100)
            f.flush()
            
            try:
                path = Path(f.name)
                # Should not raise since file is small
                DataLoader._validate_file_size(path)
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
