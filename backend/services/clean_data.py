"""
Data cleaning service with comprehensive strategies.
"""

from typing import Dict, Tuple
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
        subset: Dict[str, any] | None = None
    ) -> pd.DataFrame:
        """
        Handle duplicate rows.
        
        Args:
            df: Input dataframe
            subset: Columns to consider for duplicates
            
        Returns:
            pd.DataFrame: Data with duplicates removed
        """
        initial_rows = len(df)
        df = df.drop_duplicates(subset=subset)
        removed = initial_rows - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df

    @staticmethod
    def _handle_numerical_missing(
        df: pd.DataFrame,
        col: str,
        strategy: str = "median"
    ) -> None:
        """
        Handle missing values in numerical columns.
        
        Args:
            df: Input dataframe
            col: Column name
            strategy: Strategy - median, mean, forward_fill, backward_fill, drop
        """
        missing_count = df[col].isnull().sum()
        
        if missing_count == 0:
            return
        
        try:
            if strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "forward_fill":
                df[col].fillna(method="ffill", inplace=True)
                df[col].fillna(method="bfill", inplace=True)
            elif strategy == "backward_fill":
                df[col].fillna(method="bfill", inplace=True)
                df[col].fillna(method="ffill", inplace=True)
            elif strategy == "drop":
                df.dropna(subset=[col], inplace=True)
            else:
                logger.warning(f"Unknown strategy: {strategy}, using median")
                df[col].fillna(df[col].median(), inplace=True)
            
            logger.debug(f"Handled {missing_count} missing values in {col}")
            
        except Exception as e:
            raise DataCleaningError(
                f"Error handling missing values in {col}: {str(e)}"
            ) from e

    @staticmethod
    def _handle_categorical_missing(
        df: pd.DataFrame,
        col: str,
        strategy: str = "mode"
    ) -> None:
        """
        Handle missing values in categorical columns.
        
        Args:
            df: Input dataframe
            col: Column name
            strategy: Strategy - mode, forward_fill, backward_fill, drop
        """
        missing_count = df[col].isnull().sum()
        
        if missing_count == 0:
            return
        
        try:
            if strategy == "mode":
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col].fillna(mode_value[0], inplace=True)
                else:
                    df[col].fillna("MISSING", inplace=True)
            elif strategy == "forward_fill":
                df[col].fillna(method="ffill", inplace=True)
                df[col].fillna(method="bfill", inplace=True)
            elif strategy == "backward_fill":
                df[col].fillna(method="bfill", inplace=True)
                df[col].fillna(method="ffill", inplace=True)
            elif strategy == "drop":
                df.dropna(subset=[col], inplace=True)
            else:
                logger.warning(f"Unknown strategy: {strategy}, using mode")
                df[col].fillna("MISSING", inplace=True)
            
            logger.debug(f"Handled {missing_count} missing values in {col}")
            
        except Exception as e:
            raise DataCleaningError(
                f"Error handling missing values in {col}: {str(e)}"
            ) from e

    @staticmethod
    def _detect_and_handle_outliers(
        df: pd.DataFrame,
        col: str,
        method: str = "iqr"
    ) -> None:
        """
        Detect and handle outliers using IQR or Z-score.
        
        Args:
            df: Input dataframe
            col: Column name
            method: Method - iqr or zscore
        """
        try:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap outliers instead of removing
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    logger.debug(f"Capped {outlier_count} outliers in {col}")
                    
            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > 3
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    df.loc[outliers, col] = df[col].median()
                    logger.debug(f"Handled {outlier_count} Z-score outliers in {col}")
                    
        except Exception as e:
            logger.warning(f"Could not handle outliers in {col}: {str(e)}")

    @staticmethod
    def clean(
        df: pd.DataFrame,
        handle_outliers: bool = True,
        outlier_method: str = "iqr"
    ) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline.
        
        Args:
            df: Input dataframe
            handle_outliers: Whether to handle outliers
            outlier_method: Method for outlier detection
            
        Returns:
            pd.DataFrame: Cleaned data
            
        Raises:
            DataCleaningError: If cleaning fails
        """
        try:
            logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
            
            # Create copy to avoid modifying original
            df = df.copy()
            
            # Remove duplicates
            df = DataCleaner._handle_duplicates(df)
            
            # Separate numerical and categorical columns
            numerical_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(exclude=['number']).columns
            
            # Handle missing values
            for col in numerical_cols:
                DataCleaner._handle_numerical_missing(df, col, strategy="median")
            
            for col in categorical_cols:
                DataCleaner._handle_categorical_missing(df, col, strategy="mode")
            
            # Handle outliers in numerical columns
            if handle_outliers:
                for col in numerical_cols:
                    if df[col].notna().sum() > 0:
                        DataCleaner._detect_and_handle_outliers(
                            df, col, method=outlier_method
                        )
            
            # Remove rows with all NaN values
            df = df.dropna(how="all")
            
            logger.info(f"Data cleaning completed. Final shape: {df.shape}")
            return df
            
        except DataCleaningError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error during data cleaning: {str(e)}"
            logger.error(error_msg)
            raise DataCleaningError(error_msg) from e