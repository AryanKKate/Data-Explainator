"""
Data profiling service for exploratory data analysis.
"""

from typing import Dict, Any
import pandas as pd
import logging
from exceptions import DataValidationError

logger = logging.getLogger(__name__)


class DataProfiler:
    """Generate comprehensive data profiles."""

    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data profile.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict: Profile containing shape, columns, types, missing values, stats
            
        Raises:
            DataValidationError: If dataframe is invalid
        """
        try:
            if df is None:
                raise DataValidationError("Dataframe is None")
            if df.empty:
                raise DataValidationError("Dataframe is empty")
            if df.shape[1] == 0:
                raise DataValidationError("Dataframe has no columns")
            
            logger.info(f"Profiling data with shape: {df.shape}")
            
            profile = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing": df.isnull().sum().to_dict(),
                "missing_percentage": (
                    (df.isnull().sum() / len(df)) * 100
                ).to_dict(),
                "duplicates": int(df.duplicated().sum()),
                "duplicate_percentage": round(
                    (df.duplicated().sum() / len(df)) * 100, 2
                ),
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / 1024 / 1024, 2
                ),
                "stats": df.describe(include="all").to_dict(),
                "numeric_cols": list(
                    df.select_dtypes(include=['number']).columns
                ),
                "categorical_cols": list(
                    df.select_dtypes(exclude=['number']).columns
                ),
            }
            
            logger.debug(f"Profile generated: {len(profile)} sections")
            return profile
            
        except DataValidationError:
            raise
        except Exception as e:
            error_msg = f"Error profiling data: {str(e)}"
            logger.error(error_msg)
            raise DataValidationError(error_msg) from e