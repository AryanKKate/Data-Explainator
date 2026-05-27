"""
Validation agent for data quality checks.
"""

from typing import Dict, Any
import pandas as pd
import logging
from exceptions import DataValidationError

logger = logging.getLogger(__name__)


class ValidationAgent:
    """Validates data quality and completeness."""

    @staticmethod
    def validate(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data validation report.
        
        Args:
            df: Dataframe to validate
            
        Returns:
            Dict: Validation report with metrics
            
        Raises:
            DataValidationError: If dataframe invalid
        """
        try:
            if df is None:
                raise DataValidationError("Dataframe is None")
            if df.empty:
                raise DataValidationError("Dataframe is empty")
            
            logger.info(f"Validating data with shape: {df.shape}")
            
            # Calculate metrics
            total_cells = df.shape[0] * df.shape[1]
            missing_count = int(df.isnull().sum().sum())
            duplicates_count = int(df.duplicated().sum())
            
            report = {
                "shape": df.shape,
                "total_cells": total_cells,
                "missing_values": missing_count,
                "missing_percentage": round(
                    (missing_count / total_cells) * 100, 2
                ),
                "duplicates": duplicates_count,
                "duplicate_percentage": round(
                    (duplicates_count / len(df)) * 100 if len(df) > 0 else 0, 2
                ),
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "row_count": len(df),
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / 1024 / 1024, 2
                ),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "column_missing_values": df.isnull().sum().to_dict(),
                "quality_score": 0.0,
            }
            
            # Calculate quality score (0-100)
            quality_metrics = {
                "completeness": (1 - (missing_count / total_cells)) * 100
                if total_cells > 0 else 100,
                "uniqueness": (1 - (duplicates_count / len(df))) * 100
                if len(df) > 0 else 100,
            }
            
            report["quality_score"] = round(
                (quality_metrics["completeness"] + quality_metrics["uniqueness"]) / 2, 2
            )
            report["quality_metrics"] = quality_metrics
            
            # Add warnings
            warnings = []
            if report["missing_percentage"] > 10:
                warnings.append(f"High missing values: {report['missing_percentage']}%")
            if report["duplicate_percentage"] > 5:
                warnings.append(f"Duplicates detected: {report['duplicate_percentage']}%")
            if report["quality_score"] < 70:
                warnings.append("Low data quality score - review needed")
            
            report["warnings"] = warnings
            
            logger.info(
                f"Validation complete. Quality score: {report['quality_score']}"
            )
            return report
            
        except DataValidationError:
            raise
        except Exception as e:
            error_msg = f"Error during validation: {str(e)}"
            logger.error(error_msg)
            raise DataValidationError(error_msg) from e