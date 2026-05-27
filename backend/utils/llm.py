"""
LLM initialization and configuration.
Provides retry logic and error handling.
"""

import logging
from langchain_groq import ChatGroq
from config import config
from exceptions import LLMError, LLMRetryableError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating cached LLM instances."""

    _instance: ChatGroq | None = None

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(LLMRetryableError),
        reraise=True,
    )
    def _create_llm() -> ChatGroq:
        try:
            logger.info("Initializing LLM with model: %s", config.model_name)
            return ChatGroq(
                api_key=config.groq_api_key,
                model=config.model_name,
                temperature=config.llm_temperature,
                timeout=30,
            )
        except Exception as e:
            error_msg = f"Failed to initialize LLM: {e}"
            logger.error(error_msg)
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                raise LLMRetryableError(error_msg) from e
            raise LLMError(error_msg) from e

    @classmethod
    def get_llm(cls) -> ChatGroq:
        if cls._instance is None:
            cls._instance = cls._create_llm()
        return cls._instance


class LazyLLM:
    """Proxy object to lazily initialize LLM only when first used."""

    def __getattr__(self, item):
        return getattr(LLMFactory.get_llm(), item)


llm = LazyLLM()
