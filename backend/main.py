"""
Data Explainator - Main entry point for CLI usage.

This script demonstrates the full pipeline:
1. Load data
2. Profile and clean
3. Intent detection
4. Feature engineering
5. Model training
"""

import logging
import sys
from pathlib import Path

from logger_config import setup_logging, get_logger
from config import config
from services.load_data import DataLoader
from services.clean_data import DataCleaner
from services.profile_data import DataProfiler
from graphs.analyst_graph import app as analyst_app
from exceptions import DataExplainterException

# Setup logging
setup_logging(
    level=config.log_level,
    debug=config.debug,
)

logger = get_logger(__name__)


def main() -> int:
    """
    Main entry point for CLI.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        logger.info("=" * 60)
        logger.info("Data Explainator - AI-Powered Data Analysis Pipeline")
        logger.info("=" * 60)
        
        # Load data
        logger.info("Step 1: Loading data...")
        df = DataLoader.load("data/housing.csv")
        logger.info(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Profile data
        logger.info("\nStep 2: Profiling data...")
        profile = DataProfiler.profile(df)
        logger.info(f"✓ Data profiled")
        logger.info(f"  - Missing values: {sum(profile['missing'].values())}")
        logger.info(f"  - Duplicates: {profile.get('duplicates', 'N/A')}")
        logger.info(f"  - Memory usage: {profile['memory_usage_mb']}MB")
        
        # Clean data
        logger.info("\nStep 3: Cleaning data...")
        df_clean = DataCleaner.clean(df)
        logger.info(f"✓ Data cleaned: {df_clean.shape[0]} rows")
        
        # Run analysis pipeline
        logger.info("\nStep 4: Running analysis pipeline...")
        user_query = "Predict housing prices using price as the target variable."
        
        logger.info(f"Query: {user_query}")
        
        result = analyst_app.invoke({
            "data": df_clean,
            "profile": profile,
            "user_query": user_query,
        })
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"\nTask Type: {result.get('task_type', 'N/A')}")
        logger.info(f"Target Column: {result.get('target_column', 'N/A')}")
        
        if "validation_report" in result:
            logger.info(f"\nValidation Report:")
            for key, value in result["validation_report"].items():
                logger.info(f"  - {key}: {value}")
        
        if "model_plan" in result:
            logger.info(f"\nModel Plan:")
            logger.info(f"  - Task: {result['model_plan'].get('task')}")
            logger.info(f"  - Models: {', '.join(result['model_plan'].get('models', []))}")
        
        if "training_results" in result:
            logger.info(f"\nTraining Results:")
            for model, metrics in result["training_results"].items():
                logger.info(f"  - {model}:")
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        logger.info(f"    - {metric_name}: {metric_value}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ Analysis completed successfully!")
        logger.info("=" * 60)
        
        return 0
    
    except DataExplainterException as e:
        logger.error(f"Data Explainator Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())