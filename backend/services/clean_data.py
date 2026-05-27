"""
Data cleaning service with comprehensive strategies.
"""

from typing import Any
import pandas as pd
import numpy as np
import logging
from exceptions import DataCleaningError

logger = logging.getLogger(__name__)


class DataCleaner:
    """Data cleaning with multiple strategies and validation."""

    @staticmethod
    def _handle_duplicates(
        df: pd.DataFrame,
        subset: list[str] | None = None,
    ) -> pd.DataFrame:
        """Handle duplicate rows."""
        initial_rows = len(df)
        df = df.drop_duplicates(subset=subset)
        removed = initial_rows - len(df)

        if removed == 0 and subset is None and "id" in df.columns:
            # Industry-safe heuristic: where entity identifiers repeat,
            # keep the first observation to avoid duplicate entity records.
            deduped = df.drop_duplicates(subset=["id"], keep="first")
            removed = initial_rows - len(deduped)
            df = deduped

        if removed > 0:
            logger.info("Removed %s duplicate rows", removed)

        return df

    @staticmethod
    def _handle_numerical_missing(
        df: pd.DataFrame,
        col: str,
        strategy: str = "median",
    ) -> None:
        """Handle missing values in numerical columns."""
        missing_count = int(df[col].isnull().sum())
        if missing_count == 0:
            return

        try:
            if strategy == "median":
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
            elif strategy == "mean":
                fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
            elif strategy == "forward_fill":
                df[col] = df[col].ffill().bfill()
            elif strategy == "backward_fill":
                df[col] = df[col].bfill().ffill()
            elif strategy == "drop":
                df.dropna(subset=[col], inplace=True)
            else:
                logger.warning("Unknown strategy: %s, using median", strategy)
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)

            logger.debug("Handled %s missing values in %s", missing_count, col)

        except Exception as e:
            raise DataCleaningError(f"Error handling missing values in {col}: {e}") from e

    @staticmethod
    def _handle_categorical_missing(
        df: pd.DataFrame,
        col: str,
        strategy: str = "mode",
    ) -> None:
        """Handle missing values in categorical columns."""
        missing_count = int(df[col].isnull().sum())
        if missing_count == 0:
            return

        try:
            if strategy == "mode":
                mode_value = df[col].mode()
                fill_value = mode_value.iloc[0] if not mode_value.empty else "MISSING"
                df[col] = df[col].fillna(fill_value)
            elif strategy == "forward_fill":
                df[col] = df[col].ffill().bfill()
            elif strategy == "backward_fill":
                df[col] = df[col].bfill().ffill()
            elif strategy == "drop":
                df.dropna(subset=[col], inplace=True)
            else:
                logger.warning("Unknown strategy: %s, using MISSING", strategy)
                df[col] = df[col].fillna("MISSING")

            logger.debug("Handled %s missing values in %s", missing_count, col)

        except Exception as e:
            raise DataCleaningError(f"Error handling missing values in {col}: {e}") from e

    @staticmethod
    def _detect_and_handle_outliers(
        df: pd.DataFrame,
        col: str,
        method: str = "iqr",
    ) -> None:
        """Detect and handle outliers using IQR or Z-score."""
        try:
            original_dtype = df[col].dtype
            series = pd.to_numeric(df[col], errors="coerce").astype(float)

            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1

                if pd.isna(iqr) or iqr == 0:
                    return

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                clipped = series.clip(lower=lower_bound, upper=upper_bound)
                outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

                if outlier_count > 0:
                    df[col] = clipped.astype(original_dtype, copy=False)
                    logger.debug("Capped %s outliers in %s", outlier_count, col)

            elif method == "zscore":
                std = series.std()
                if std == 0 or pd.isna(std):
                    return
                z_scores = np.abs((series - series.mean()) / std)
                outliers = z_scores > 3
                outlier_count = int(outliers.sum())

                if outlier_count > 0:
                    series.loc[outliers] = series.median()
                    df[col] = series.astype(original_dtype, copy=False)
                    logger.debug("Handled %s Z-score outliers in %s", outlier_count, col)

        except Exception as e:
            logger.warning("Could not handle outliers in %s: %s", col, e)

    @staticmethod
    def clean(
        df: pd.DataFrame,
        handle_outliers: bool = True,
        outlier_method: str = "iqr",
    ) -> pd.DataFrame:
        """Comprehensive data cleaning pipeline."""
        try:
            if df is None or not isinstance(df, pd.DataFrame):
                raise DataCleaningError("Input must be a pandas DataFrame")
            if df.empty:
                raise DataCleaningError("Input DataFrame is empty")

            logger.info("Starting data cleaning. Initial shape: %s", df.shape)
            df = df.copy()

            df = DataCleaner._handle_duplicates(df)
            numerical_cols = df.select_dtypes(include=["number"]).columns
            categorical_cols = df.select_dtypes(exclude=["number"]).columns

            for col in numerical_cols:
                DataCleaner._handle_numerical_missing(df, col, strategy="median")

            for col in categorical_cols:
                DataCleaner._handle_categorical_missing(df, col, strategy="mode")

            if handle_outliers:
                for col in numerical_cols:
                    if df[col].notna().any():
                        DataCleaner._detect_and_handle_outliers(df, col, method=outlier_method)

            df = df.dropna(how="all")

            logger.info("Data cleaning completed. Final shape: %s", df.shape)
            return df

        except DataCleaningError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error during data cleaning: {e}"
            logger.error(error_msg)
            raise DataCleaningError(error_msg) from e
