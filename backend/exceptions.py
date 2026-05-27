"""
Custom exceptions for the application.
Provides specific error handling.
"""


class DataExplainterException(Exception):
    """Base exception for the application."""
    pass


class FileOperationError(DataExplainterException):
    """Raised when file operations fail."""
    pass


class FileNotSupportedError(FileOperationError):
    """Raised when file format is not supported."""
    pass


class FileSizeExceededError(FileOperationError):
    """Raised when file size exceeds limit."""
    pass


class DataValidationError(DataExplainterException):
    """Raised when data validation fails."""
    pass


class DataCleaningError(DataExplainterException):
    """Raised when data cleaning fails."""
    pass


class LLMError(DataExplainterException):
    """Raised when LLM operations fail."""
    pass


class LLMRetryableError(LLMError):
    """Raised when LLM fails with a retryable error."""
    pass


class ModelTrainingError(DataExplainterException):
    """Raised when model training fails."""
    pass


class ConfigurationError(DataExplainterException):
    """Raised when configuration is invalid."""
    pass


class SchemaError(DataExplainterException):
    """Raised when schema validation fails."""
    pass


class FeatureEngineeringError(DataExplainterException):
    """Raised when feature engineering fails."""
    pass
