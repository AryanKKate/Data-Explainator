"""
Feature engineering agent for data transformation and preprocessing.
Applies various feature engineering techniques.
"""

from typing import Dict, List, Any, Optional
import logging
import pandas as pd
from tools.preprocessing_tools import FeatureTools
from exceptions import FeatureEngineeringError, DataValidationError

logger = logging.getLogger(__name__)


class FeatureAgent:
    """Applies feature engineering transformations."""

    # Available preprocessing steps
    AVAILABLE_STEPS = {
        "encoding",
        "scaling",
        "handle_imbalance",
        "handle_outliers",
        "date_features",
        "lagging",
        "normalization",
    }

    @staticmethod
    def process(
        df: pd.DataFrame,
        steps: List[str],
        schema: Dict[str, Any],
        semantic_schema: Dict[str, Any],
        target: str,
    ) -> pd.DataFrame:
        """
        Apply feature engineering pipeline.
        
        Args:
            df: Input dataframe
            steps: List of preprocessing steps
            schema: Column schema
            semantic_schema: Semantic information
            target: Target column name
            
        Returns:
            pd.DataFrame: Transformed data
            
        Raises:
            FeatureEngineeringError: If engineering fails
            DataValidationError: If inputs invalid
        """
        try:
            # Validate inputs
            if df is None or df.empty:
                raise DataValidationError("Dataframe cannot be None or empty")
            
            if not isinstance(steps, list) or not steps:
                raise DataValidationError("Steps must be a non-empty list")
            
            if not isinstance(schema, dict):
                raise DataValidationError("Schema must be a dict")
            
            if not isinstance(semantic_schema, dict):
                raise DataValidationError("Semantic schema must be a dict")
            
            if target and target not in df.columns:
                logger.warning(f"Target column '{target}' not found in dataframe")
            
            logger.info(
                f"Starting feature engineering with {len(steps)} steps "
                f"on data shape: {df.shape}"
            )
            
            # Create copy to avoid modifying original
            df = df.copy()
            
            # Step 1: Remove identifier columns
            try:
                df = FeatureTools.remove_ids(df, schema)
                logger.debug(f"After ID removal: {df.shape}")
            except Exception as e:
                logger.error(f"Error removing IDs: {e}")
                raise FeatureEngineeringError(f"ID removal failed: {e}") from e
            
            # Step 2: Process semantic values
            try:
                df = FeatureTools.process_semantic_values(df, semantic_schema)
                logger.debug(f"After semantic processing: {df.shape}")
            except Exception as e:
                logger.error(f"Error processing semantic values: {e}")
                raise FeatureEngineeringError(f"Semantic processing failed: {e}") from e
            
            # Step 3: Convert semantic numeric
            try:
                df = FeatureTools.convert_semantic_numeric(df, semantic_schema)
                logger.debug(f"After semantic numeric conversion: {df.shape}")
            except Exception as e:
                logger.warning(f"Error converting semantic numeric: {e}")
                # Don't fail on this
            
            # Step 4: Process dates
            try:
                df = FeatureTools.process_dates(df, schema)
                logger.debug(f"After date processing: {df.shape}")
            except Exception as e:
                logger.warning(f"Error processing dates: {e}")
                # Don't fail on this
            
            # Step 5: Process cyclical features
            try:
                df = FeatureTools.process_cyclical_features(df)
                logger.debug(f"After cyclical processing: {df.shape}")
            except Exception as e:
                logger.warning(f"Error processing cyclical features: {e}")
                # Don't fail on this
            
            # Step 6: Handle missing values
            try:
                df = FeatureTools.handle_missing(df)
                logger.debug(f"After missing handling: {df.shape}")
            except Exception as e:
                logger.error(f"Error handling missing values: {e}")
                raise FeatureEngineeringError(f"Missing value handling failed: {e}") from e
            
            # Step 7: Normalize categories
            try:
                df = FeatureTools.normalize_categories(df)
                logger.debug(f"After category normalization: {df.shape}")
            except Exception as e:
                logger.warning(f"Error normalizing categories: {e}")
                # Don't fail on this
            
            # Step 8: Encode target
            if target and target in df.columns:
                try:
                    df = FeatureTools.encode_target(df, target)
                    logger.debug(f"After target encoding: {df.shape}")
                except Exception as e:
                    logger.warning(f"Error encoding target: {e}")
                    # Don't fail on this
            
            # Step 9: Apply requested preprocessing steps
            for step in steps:
                step = step.lower().strip()
                
                if step not in FeatureAgent.AVAILABLE_STEPS:
                    logger.warning(f"Unknown step: {step}, skipping")
                    continue
                
                try:
                    if step == "encoding":
                        logger.info("Applying encoding...")
                        df = FeatureTools.encode(df, target)
                    
                    elif step == "scaling":
                        logger.info("Applying scaling...")
                        df = FeatureTools.scale(df, target, schema)
                    
                    elif step == "handle_imbalance":
                        logger.info("Handling imbalance...")
                        if target and target in df.columns:
                            try:
                                df = FeatureTools.handle_imbalance(df, target)
                            except Exception as e:
                                logger.warning(f"Could not handle imbalance: {e}")
                    
                    elif step == "handle_outliers":
                        logger.info("Handling outliers...")
                        df = FeatureTools.handle_outliers(df)
                    
                    elif step == "date_features":
                        logger.info("Creating date features...")
                        df = FeatureTools.create_date_features(df)
                    
                    elif step == "lagging":
                        logger.info("Creating lagged features...")
                        df = FeatureTools.create_lag_features(df)
                    
                    elif step == "normalization":
                        logger.info("Normalizing...")
                        df = FeatureTools.normalize(df)
                    
                    logger.debug(f"After {step}: {df.shape}")
                    
                except Exception as e:
                    logger.error(f"Error in step '{step}': {e}")
                    raise FeatureEngineeringError(
                        f"Step '{step}' failed: {str(e)}"
                    ) from e
            
            logger.info(f"Feature engineering complete. Final shape: {df.shape}")
            
            # Validate output
            if df.empty:
                raise FeatureEngineeringError("Feature engineering resulted in empty dataframe")
            
            if df.isnull().all().any():
                logger.warning("Some columns are entirely null after feature engineering")
            
            return df
        
        except (FeatureEngineeringError, DataValidationError):
            raise
        except Exception as e:
            error_msg = f"Unexpected error in feature engineering: {str(e)}"
            logger.error(error_msg)
            raise FeatureEngineeringError(error_msg) from e
