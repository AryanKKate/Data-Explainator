"""
Schema analysis agent for determining column roles.
"""

from typing import Dict, Literal, List
import logging
from pydantic import BaseModel, Field, validator
from utils.llm import llm
from exceptions import LLMError, SchemaError

logger = logging.getLogger(__name__)

ColumnRole = Literal[
    "identifier",
    "numerical",
    "categorical",
    "datetime",
    "target",
    "unknown",
]


class SchemaOutput(BaseModel):
    """Structured output for schema analysis."""
    
    column_roles: Dict[str, ColumnRole] = Field(
        ...,
        description="Mapping of column names to their roles"
    )
    
    @validator("column_roles")
    def not_empty(cls, v: Dict) -> Dict:
        """Validate column_roles is not empty."""
        if not v:
            raise ValueError("column_roles cannot be empty")
        return v


class SchemaAgent:
    """Analyzes data schema and determines column roles."""

    ROLE_KEYWORDS = {
        "identifier": ["id", "uuid", "customer", "user", "pk", "key"],
        "datetime": ["date", "time", "timestamp", "created", "updated", "at"],
        "target": ["target", "label", "outcome", "y", "churn", "class"],
    }

    @staticmethod
    def analyze(columns: List[str]) -> Dict:
        """
        Analyze columns and determine their roles.
        
        Args:
            columns: List of column names
            
        Returns:
            Dict: Mapping of column names to roles
            
        Raises:
            SchemaError: If analysis fails
        """
        try:
            # Validate input
            if not columns or not isinstance(columns, list):
                raise SchemaError("Columns must be a non-empty list")
            
            if not all(isinstance(c, str) for c in columns):
                raise SchemaError("All column names must be strings")
            
            logger.info(f"Analyzing schema for {len(columns)} columns")
            
            # Create structured LLM
            structured_llm = llm.with_structured_output(SchemaOutput)
            
            # Craft prompt
            prompt = f"""
Analyze the following columns and determine the role of each.

Columns:
{columns}

Determine role for EACH column. Possible roles:
- identifier: ID, customer ID, user ID, primary key
- numerical: Numeric measurements or quantities
- categorical: Text/discrete values (categories, regions, types)
- datetime: Date, time, timestamp values
- target: The column to predict or analyze
- unknown: Cannot determine

Examples:
- customer_id, id, user_id → identifier
- price, age, amount → numerical
- city, gender, status → categorical
- date, timestamp, last_login → datetime
- churn, target, label, price (if predicting) → target

Return the role for EACH column provided.
Make sure to include all columns in your response.
"""
            
            # Call LLM
            try:
                result = structured_llm.invoke(prompt)
                output = result.model_dump()
                
                # Validate we have all columns
                if len(output["column_roles"]) != len(columns):
                    logger.warning(
                        f"LLM returned {len(output['column_roles'])} roles "
                        f"but got {len(columns)} columns"
                    )
                    # Add missing columns as unknown
                    for col in columns:
                        if col not in output["column_roles"]:
                            output["column_roles"][col] = "unknown"
                
                logger.info(f"Schema analysis complete")
                return output
                
            except Exception as e:
                logger.warning(
                    f"LLM schema analysis failed, using heuristic: {e}"
                )
                # Fallback to heuristic analysis
                return SchemaAgent._analyze_heuristic(columns)
            
        except SchemaError:
            raise
        except Exception as e:
            error_msg = f"Error analyzing schema: {str(e)}"
            logger.error(error_msg)
            raise SchemaError(error_msg) from e

    @staticmethod
    def _analyze_heuristic(columns: List[str]) -> Dict:
        """
        Fallback heuristic schema analysis.
        
        Args:
            columns: List of column names
            
        Returns:
            Dict: Mapping of columns to roles
        """
        logger.debug("Using heuristic schema analysis")
        column_roles = {}
        
        for col in columns:
            col_lower = col.lower()
            role = "unknown"
            
            # Check identifier keywords
            if any(kw in col_lower for kw in SchemaAgent.ROLE_KEYWORDS["identifier"]):
                role = "identifier"
            # Check datetime keywords
            elif any(kw in col_lower for kw in SchemaAgent.ROLE_KEYWORDS["datetime"]):
                role = "datetime"
            # Check target keywords
            elif any(kw in col_lower for kw in SchemaAgent.ROLE_KEYWORDS["target"]):
                role = "target"
            # Default to categorical
            else:
                role = "categorical"
            
            column_roles[col] = role
        
        return {"column_roles": column_roles}