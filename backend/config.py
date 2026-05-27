"""
Configuration management with validation.
Follows the 12-factor app principles.
"""

import os
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is invalid."""


class Config:
    """Application configuration with lazy validation."""

    def __init__(self) -> None:
        load_dotenv()

    def validate_llm_config(self) -> None:
        """Validate only the settings required for LLM usage."""
        if not os.getenv("GROQ_API_KEY"):
            raise ConfigError("GROQ_API_KEY not set")

    @property
    def groq_api_key(self) -> str:
        self.validate_llm_config()
        return os.getenv("GROQ_API_KEY", "")

    @property
    def model_name(self) -> str:
        return os.getenv("MODEL_NAME", "mixtral-8x7b-32768")

    @property
    def llm_temperature(self) -> float:
        try:
            return float(os.getenv("LLM_TEMPERATURE", "0.2"))
        except ValueError as e:
            raise ConfigError("LLM_TEMPERATURE must be a float") from e

    @property
    def max_file_size_mb(self) -> int:
        try:
            return int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        except ValueError as e:
            raise ConfigError("MAX_FILE_SIZE_MB must be an integer") from e

    @property
    def debug(self) -> bool:
        return os.getenv("DEBUG", "False").lower() == "true"

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")

    @property
    def test_mode(self) -> bool:
        return os.getenv("TEST_MODE", "False").lower() == "true"


config = Config()
