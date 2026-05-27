"""
LLM initialization and configuration.
Provides retry logic and error handling.
"""

from typing import Optional
import logging
from langchain_groq import ChatGroq
from config import config
from exceptions import LLMError, LLMRetryableError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(LLMRetryableError),
        reraise=True,
    )
    def get_llm() -> ChatGroq:
        """
        Get or create LLM instance with retry logic.
        
        Returns:
            ChatGroq: Configured LLM instance
            
        Raises:
            LLMError: If LLM initialization fails
        """
        try:
            logger.info(f"Initializing LLM with model: {config.model_name}")
            
            llm_instance = ChatGroq(
                api_key=config.groq_api_key,
                model=config.model_name,
                temperature=config.llm_temperature,
                timeout=30,
            )
            
            logger.info("LLM initialized successfully")
            return llm_instance
            
        except Exception as e:
            error_msg = f"Failed to initialize LLM: {str(e)}"
            logger.error(error_msg)
            
            # Determine if error is retryable
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                raise LLMRetryableError(error_msg) from e
            raise LLMError(error_msg) from e


# Initialize global LLM instance with error handling
try:
    llm = LLMFactory.get_llm()
    logger.info("Global LLM instance created")
except LLMError as e:
    logger.error(f"Fatal: Could not initialize LLM: {e}")
    raise