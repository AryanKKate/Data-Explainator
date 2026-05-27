"""
Intent detection agent for identifying ML task type.
"""

from typing import Dict, Literal
import logging
from pydantic import BaseModel, Field, validator
from utils.llm import llm
from exceptions import LLMError, DataValidationError

logger = logging.getLogger(__name__)

TaskType = Literal[
    "classification",
    "regression",
    "clustering",
    "forecasting",
    "anomaly_detection",
    "exploratory_analysis",
]


class IntentOutput(BaseModel):
    """Structured output for intent detection."""
    
    task_type: TaskType = Field(
        ..., 
        description="The detected ML task type"
    )
    target: str = Field(
        ..., 
        description="The target column name"
    )
    confidence: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Confidence score of detection"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of detection"
    )
    
    @validator("target")
    def target_not_empty(cls, v: str) -> str:
        """Validate target is not empty."""
        if not v or not v.strip():
            raise ValueError("Target column cannot be empty")
        return v.strip()


class IntentAgent:
    """Detects the ML task type from user query."""

    @staticmethod
    def detect(query: str) -> Dict:
        """
        Detect ML task type from user query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dict: Task type, target column, confidence, and reasoning
            
        Raises:
            DataValidationError: If query is invalid
            LLMError: If LLM call fails
        """
        try:
            # Validate input
            if not query or not isinstance(query, str):
                raise DataValidationError("Query must be a non-empty string")
            
            if len(query.strip()) < 5:
                raise DataValidationError(
                    "Query too short. Provide more context (min 5 chars)"
                )
            
            logger.info(f"Detecting intent for query: {query[:50]}...")
            
            # Create structured LLM
            structured_llm = llm.with_structured_output(IntentOutput)
            
            # Craft prompt
            prompt = f"""
Analyze the following user query and determine:
1. The ML task type (classification, regression, clustering, forecasting, anomaly_detection, or exploratory_analysis)
2. The target column (what should be predicted/analyzed)
3. Your confidence score (0-1)
4. Brief reasoning

Task categories:
- Classification: Predicting discrete/categorical outcomes
- Regression: Predicting continuous numerical values
- Clustering: Grouping similar data points
- Forecasting: Time-series prediction
- Anomaly Detection: Finding unusual patterns
- Exploratory Analysis: Understanding data without specific target

User Query:
{query}

Be precise. If target is unclear, make your best inference.
"""
            
            # Call LLM
            result = structured_llm.invoke(prompt)
            
            # Convert to dict
            output = result.model_dump()
            
            logger.info(
                f"Intent detected: {output['task_type']} "
                f"(confidence: {output['confidence']})"
            )
            return output
            
        except DataValidationError:
            raise
        except LLMError:
            raise
        except Exception as e:
            error_msg = f"Error detecting intent: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e