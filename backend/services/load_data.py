"""
Data loading service with validation and error handling.
"""

from typing import Union
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
import logging
from config import config
from exceptions import (
    FileOperationError,
    FileNotSupportedError,
    FileSizeExceededError,
    DataValidationError,
)

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json"}


class DataLoader:
    """Load data from various file formats with validation."""

    @staticmethod
    def _validate_file_path(file_path: str) -> Path:
        """
        Validate file path exists and is accessible.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Path: Validated Path object
            
        Raises:
            FileOperationError: If file doesn't exist
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileOperationError(f"File not found: {file_path}")
            if not path.is_file():
                raise FileOperationError(f"Path is not a file: {file_path}")
            return path
        except FileNotFoundError as e:
            raise FileOperationError(f"Cannot access file: {file_path}") from e

    @staticmethod
    def _validate_file_format(file_path: Path) -> None:
        """
        Validate file format is supported.
        
        Args:
            file_path: Path to the file
            
        Raises:
            FileNotSupportedError: If format not supported
        """
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise FileNotSupportedError(
                f"Unsupported file format: {suffix}. "
                f"Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

    @staticmethod
    def _validate_file_size(file_path: Path) -> None:
        """
        Validate file size doesn't exceed limit.
        
        Args:
            file_path: Path to the file
            
        Raises:
            FileSizeExceededError: If file too large
        """
        max_bytes = config.max_file_size_mb * 1024 * 1024
        file_size = file_path.stat().st_size
        
        if file_size > max_bytes:
            raise FileSizeExceededError(
                f"File size {file_size / 1024 / 1024:.1f}MB exceeds limit "
                f"of {config.max_file_size_mb}MB"
            )

    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        """
        Load data from file with full validation.
        
        Args:
            file_path: Path to data file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileOperationError: If file operations fail
            FileNotSupportedError: If format not supported
            FileSizeExceededError: If file too large
            DataValidationError: If data invalid
        """
        try:
            logger.info(f"Loading data from: {file_path}")
            
            # Validate file
            path = DataLoader._validate_file_path(file_path)
            DataLoader._validate_file_format(path)
            DataLoader._validate_file_size(path)
            
            # Load based on format
            suffix = path.suffix.lower()
            
            if suffix == ".csv":
                df = pd.read_csv(file_path)
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(file_path)
            elif suffix == ".json":
                df = pd.read_json(file_path)
            else:
                raise FileNotSupportedError(f"Unexpected format: {suffix}")
            
            # Validate loaded data
            if df.empty:
                raise DataValidationError("Loaded data is empty")
            if df.shape[1] == 0:
                raise DataValidationError("Data has no columns")
            
            logger.info(
                f"Data loaded successfully: {df.shape[0]} rows, "
                f"{df.shape[1]} columns"
            )
            return df
            
        except (FileOperationError, FileNotSupportedError, FileSizeExceededError, DataValidationError):
            raise
        except EmptyDataError as e:
            raise DataValidationError("Loaded data is empty or has no columns") from e
        except Exception as e:
            error_msg = f"Error loading file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise FileOperationError(error_msg) from e