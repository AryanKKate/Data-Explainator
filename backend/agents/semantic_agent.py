"""
Semantic analysis agent for understanding column semantics.
Detects column meanings, units, and data types.
"""

from typing import Dict, Any, List, Literal, Optional
import pandas as pd
import numpy as np
import re
import logging
from exceptions import DataValidationError

logger = logging.getLogger(__name__)

SemanticType = Literal[
    "identifier",
    "location",
    "area",
    "currency",
    "distance",
    "percentage",
    "datetime",
    "boolean",
    "numeric",
    "categorical",
    "unknown",
]


class SemanticAgent:
    """Analyzes semantic meaning of columns."""

    # Semantic patterns for column name matching
    SEMANTIC_PATTERNS = {
        "identifier": [
            "id", "uuid", "customer", "user", "order",
            "transaction", "house", "pk", "key"
        ],
        "location": [
            "city", "country", "state", "region",
            "locality", "district", "address", "zip", "postal"
        ],
        "area": [
            "area", "sqft", "square", "acre", "m2", "m²"
        ],
        "currency": [
            "price", "salary", "income", "cost", "expense",
            "maintenance", "revenue", "profit", "amount", "rate"
        ],
        "distance": [
            "distance", "km", "mile", "meter", "length", "range"
        ],
        "percentage": [
            "percent", "percentage", "ratio", "rate", "pct"
        ],
        "datetime": [
            "date", "time", "day", "month", "year",
            "timestamp", "created", "updated", "at"
        ],
    }

    # Boolean tokens
    BOOLEAN_TOKENS = {
        "yes", "no", "true", "false", "y", "n", "0", "1",
        "yes", "no", "enabled", "disabled"
    }

    # Unit patterns for regex matching
    UNIT_PATTERNS = {
        r"km": "distance",
        r"mile": "distance",
        r"meter": "distance",
        r"sq\.?ft": "area",
        r"sq\.?m": "area",
        r"acre": "area",
        r"₹|\$|€|£": "currency",
        r"%": "percentage",
    }

    @staticmethod
    def _analyze_numeric_column(series: pd.Series) -> Dict[str, Any]:
        """
        Analyze numeric column characteristics.
        
        Args:
            series: Column data
            
        Returns:
            Dict: Column info
        """
        try:
            unique_ratio = series.nunique() / len(series)
            
            if unique_ratio < 0.1:
                semantic = "ordinal"
            elif unique_ratio < 0.4:
                semantic = "categorical"
            else:
                semantic = "continuous"
            
            return {
                "type": "numeric",
                "semantic": semantic,
                "confidence": 0.9,
                "stats": {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            }
        except Exception as e:
            logger.warning(f"Error analyzing numeric column: {e}")
            return {
                "type": "numeric",
                "semantic": "unknown",
                "confidence": 0.5,
            }

    @staticmethod
    def _analyze_datetime_column(series: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Analyze if column is datetime.
        
        Args:
            series: Column data
            
        Returns:
            Dict or None: Column info if datetime else None
        """
        try:
            # Try parsing as datetime
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            parse_ratio = parsed.notna().mean()
            
            if parse_ratio > 0.8:
                return {
                    "type": "datetime",
                    "semantic": "timestamp",
                    "confidence": 0.95,
                }
            return None
        except Exception as e:
            logger.debug(f"Could not parse as datetime: {e}")
            return None

    @staticmethod
    def _analyze_boolean_column(values: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Analyze if column is boolean.
        
        Args:
            values: Column values as strings
            
        Returns:
            Dict or None: Column info if boolean else None
        """
        try:
            unique_vals = set(values.dropna().str.lower().unique())
            
            if unique_vals.issubset(SemanticAgent.BOOLEAN_TOKENS):
                return {
                    "type": "boolean",
                    "semantic": "binary",
                    "confidence": 0.95,
                }
            return None
        except Exception:
            return None

    @staticmethod
    def _detect_semantic_from_units(
        text: str,
        patterns: Dict[str, str]
    ) -> Optional[tuple]:
        """
        Detect semantic meaning from unit patterns.
        
        Args:
            text: Text to search
            patterns: Regex patterns mapping to semantics
            
        Returns:
            Tuple of (semantic, pattern) or None
        """
        for pattern, semantic in patterns.items():
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    return (semantic, pattern)
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")
        return None

    @staticmethod
    def _detect_semantic_from_keywords(
        col_name: str,
        col_text: str,
        patterns: Dict[str, List[str]]
    ) -> Optional[str]:
        """
        Detect semantic meaning from keywords.
        
        Args:
            col_name: Column name
            col_text: Sample column text
            patterns: Keyword patterns by semantic
            
        Returns:
            Semantic type or None
        """
        # Priority order for keyword matching
        priority = [
            "identifier", "currency", "distance", "area",
            "percentage", "location", "datetime"
        ]
        
        tokens = col_name.lower().split("_")
        
        for semantic in priority:
            keywords = patterns.get(semantic, [])
            if any(keyword in tokens for keyword in keywords):
                return semantic
        
        return None

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive semantic analysis of dataframe columns.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict: Mapping of column names to semantic info
            
        Raises:
            DataValidationError: If dataframe invalid
        """
        try:
            if df is None:
                raise DataValidationError("Dataframe is None")
            if df.empty:
                raise DataValidationError("Dataframe is empty")
            
            logger.info(f"Starting semantic analysis for {len(df.columns)} columns")
            
            semantic_schema = {}
            
            for col in df.columns:
                try:
                    # Sample values for analysis
                    values = df[col].dropna().astype(str).head(30)
                    
                    if len(values) == 0:
                        logger.warning(f"Column {col} has no non-null values")
                        semantic_schema[col] = {
                            "type": "unknown",
                            "semantic": "unknown",
                            "unit": None,
                            "nullable": True,
                            "confidence": 0.0,
                        }
                        continue
                    
                    joined_text = " ".join(values).lower()
                    col_lower = col.lower()
                    
                    # Initialize info dict
                    info = {
                        "type": "unknown",
                        "semantic": "unknown",
                        "unit": None,
                        "nullable": df[col].isnull().any(),
                        "confidence": 0.0,
                    }
                    
                    # Check if numeric
                    numeric_ratio = pd.to_numeric(
                        df[col], errors="coerce"
                    ).notna().mean()
                    
                    if numeric_ratio > 0.7:
                        info.update(SemanticAgent._analyze_numeric_column(df[col]))
                    
                    # Check if datetime
                    if numeric_ratio < 0.7:
                        datetime_info = SemanticAgent._analyze_datetime_column(df[col])
                        if datetime_info:
                            info.update(datetime_info)
                    
                    # Check if boolean
                    if info["semantic"] == "unknown" and numeric_ratio < 0.7:
                        bool_info = SemanticAgent._analyze_boolean_column(values)
                        if bool_info:
                            info.update(bool_info)
                    
                    # Detect units
                    if info["semantic"] == "unknown":
                        unit_result = SemanticAgent._detect_semantic_from_units(
                            joined_text, SemanticAgent.UNIT_PATTERNS
                        )
                        if unit_result:
                            semantic, pattern = unit_result
                            info["semantic"] = semantic
                            info["unit"] = pattern
                            info["confidence"] = 0.95
                    
                    # Detect from keywords
                    if info["semantic"] == "unknown":
                        semantic = SemanticAgent._detect_semantic_from_keywords(
                            col_lower, joined_text, SemanticAgent.SEMANTIC_PATTERNS
                        )
                        if semantic:
                            info["semantic"] = semantic
                            info["confidence"] = 0.95
                            info["type"] = "categorical"
                    
                    semantic_schema[col] = info
                    logger.debug(
                        f"Column {col}: {info['semantic']} "
                        f"(confidence: {info['confidence']})"
                    )
                    
                except Exception as e:
                    logger.error(f"Error analyzing column {col}: {e}")
                    semantic_schema[col] = {
                        "type": "unknown",
                        "semantic": "unknown",
                        "unit": None,
                        "nullable": True,
                        "confidence": 0.0,
                        "error": str(e),
                    }
            
            logger.info("Semantic analysis complete")
            return semantic_schema
            
        except DataValidationError:
            raise
        except Exception as e:
            error_msg = f"Error during semantic analysis: {str(e)}"
            logger.error(error_msg)
            raise DataValidationError(error_msg) from e
