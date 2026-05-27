"""
Planning agent for creating data preprocessing strategies.
"""

from typing import Dict, List, Literal
import logging
from pydantic import BaseModel, Field
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


class PlanOutput(BaseModel):
    """Structured output for plan creation."""
    
    steps: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of preprocessing steps"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of chosen steps"
    )


class PlanningAgent:
    """Creates data preprocessing and modeling plans."""

    # Predefined strategies by task type
    STRATEGIES = {
        "classification": [
            "encoding",
            "scaling",
            "handle_imbalance",
        ],
        "regression": [
            "encoding",
            "scaling",
            "handle_outliers",
        ],
        "clustering": [
            "scaling",
            "handle_missing_values",
        ],
        "forecasting": [
            "date_features",
            "lagging",
            "scaling",
        ],
        "anomaly_detection": [
            "scaling",
            "normalization",
        ],
        "exploratory_analysis": [
            "basic_cleaning",
            "profiling",
        ],
    }

    @staticmethod
    def create_plan(task_type: str) -> Dict:
        """
        Create preprocessing plan for given task type.
        
        Args:
            task_type: ML task type (classification, regression, etc.)
            
        Returns:
            Dict: Plan with steps and reasoning
            
        Raises:
            DataValidationError: If task_type is invalid
            LLMError: If LLM call fails
        """
        try:
            # Validate input
            if not task_type or not isinstance(task_type, str):
                raise DataValidationError(
                    "Task type must be a non-empty string"
                )
            
            task_type = task_type.lower().strip()
            
            if task_type not in PlanningAgent.STRATEGIES:
                valid_types = ", ".join(PlanningAgent.STRATEGIES.keys())
                raise DataValidationError(
                    f"Invalid task type: {task_type}. "
                    f"Valid types: {valid_types}"
                )
            
            logger.info(f"Creating plan for task: {task_type}")
            
            # Get base strategy
            base_steps = PlanningAgent.STRATEGIES[task_type]
            
            # Create structured LLM
            structured_llm = llm.with_structured_output(PlanOutput)
            
            # Craft prompt with specific requirements
            prompt = f"""
You are a data science expert creating a preprocessing pipeline.

Task Type: {task_type}

Base Steps: {', '.join(base_steps)}

Provide exactly the preprocessing steps needed for this task.

Rules:
- For Classification: Always include encoding, scaling, and imbalance handling
- For Regression: Include encoding, scaling, and outlier handling
- For Clustering: Include scaling and missing value handling
- For Forecasting: Include date feature extraction and lagging
- For Anomaly Detection: Include scaling and normalization
- For Exploratory Analysis: Include basic cleaning and profiling

Do NOT add extra steps beyond what's needed.
Return steps as a list of action names (e.g., "encoding", "scaling", "handle_imbalance").
"""
            
            # Call LLM with fallback
            try:
                result = structured_llm.invoke(prompt)
                output = result.model_dump()
            except Exception as e:
                logger.warning(
                    f"LLM-based plan creation failed, using default strategy: {e}"
                )
                # Fallback to predefined strategy
                output = {
                    "steps": base_steps,
                    "reasoning": f"Default strategy for {task_type}"
                }
            
            logger.info(f"Plan created with {len(output['steps'])} steps")
            return output
            
        except DataValidationError:
            raise
        except LLMError:
            raise
        except Exception as e:
            error_msg = f"Error creating plan: {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e

        return result.model_dump()