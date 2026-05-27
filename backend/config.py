"""
Configuration management with validation.
Follows the 12-factor app principles.
"""

import os
from typing import Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


class Config:
    """Application configuration with validation."""

    def __init__(self) -> None:
        load_dotenv()
        self._validate()

    def _validate(self) -> None:
        """Validate all required environment variables."""
        required_vars = ["GROQ_API_KEY", "MODEL_NAME"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ConfigError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    @property
    def groq_api_key(self) -> str:
        """Get Groq API key."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ConfigError("GROQ_API_KEY not set")
        return api_key

    @property
    def model_name(self) -> str:
        """Get model name."""
        model = os.getenv("MODEL_NAME", "mixtral-8x7b-32768")
        return model

    @property
    def llm_temperature(self) -> float:
        """Get LLM temperature."""
        try:
            return float(os.getenv("LLM_TEMPERATURE", "0.2"))
        except ValueError:
            raise ConfigError("LLM_TEMPERATURE must be a float")

    @property
    def max_file_size_mb(self) -> int:
        """Get max file size in MB."""
        try:
            return int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        except ValueError:
            raise ConfigError("MAX_FILE_SIZE_MB must be an integer")

    @property
    def debug(self) -> bool:
        """Get debug mode."""
        return os.getenv("DEBUG", "False").lower() == "true"

    @property
    def log_level(self) -> str:
        """Get log level."""
        return os.getenv("LOG_LEVEL", "INFO")

    @property
    def test_mode(self) -> bool:
        """Get test mode."""
        return os.getenv("TEST_MODE", "False").lower() == "true"


# Global config instance
config = Config()
