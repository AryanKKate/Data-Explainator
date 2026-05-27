"""
Model selection agent for choosing appropriate ML models.
"""

from typing import Dict, List, Literal, Any
import logging
import pandas as pd
from exceptions import DataValidationError, SchemaError

logger = logging.getLogger(__name__)

TaskType = Literal[
    "classification",
    "regression",
    "clustering",
    "forecasting",
    "anomaly_detection",
    "exploratory_analysis",
]

# Model recommendations by task type
MODEL_RECOMMENDATIONS = {
    "classification": {
        "models": ["random_forest", "xgboost", "lightgbm", "catboost", "logistic_regression"],
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"],
        "default_model": "xgboost",
    },
    "regression": {
        "models": ["random_forest", "xgboost", "lightgbm", "catboost", "linear_regression"],
        "metrics": ["rmse", "mae", "r2", "mape"],
        "default_model": "xgboost",
    },
    "clustering": {
        "models": ["kmeans", "dbscan", "hierarchical", "gaussian_mixture"],
        "metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
        "default_model": "kmeans",
    },
    "forecasting": {
        "models": ["arima", "exponential_smoothing", "prophet", "lstm"],
        "metrics": ["mae", "rmse", "mape"],
        "default_model": "exponential_smoothing",
    },
    "anomaly_detection": {
        "models": ["isolation_forest", "local_outlier_factor", "elliptic_envelope"],
        "metrics": ["precision", "recall", "f1", "roc_auc"],
        "default_model": "isolation_forest",
    },
    "exploratory_analysis": {
        "models": ["descriptive_stats", "visualization", "pca"],
        "metrics": ["variance_explained"],
        "default_model": "descriptive_stats",
    },
}


class ModelSelectionAgent:
    """Selects appropriate models based on task and data characteristics."""

    @staticmethod
    def select(
        task_type: str,
        schema: Dict[str, Any],
        df: pd.DataFrame,
        target_col: str,
    ) -> Dict[str, Any]:
        """
        Select appropriate models for the task.
        
        Args:
            task_type: ML task type
            schema: Schema information
            df: Input dataframe
            target_col: Target column name
            
        Returns:
            Dict: Model recommendations with rationale
            
        Raises:
            DataValidationError: If inputs invalid
            SchemaError: If schema invalid
        """
        try:
            # Validate inputs
            if not task_type or not isinstance(task_type, str):
                raise DataValidationError("Task type must be a non-empty string")
            
            if df is None or df.empty:
                raise DataValidationError("Dataframe cannot be None or empty")
            
            if not target_col or not isinstance(target_col, str):
                raise DataValidationError("Target column must be a non-empty string")
            
            if target_col not in df.columns:
                raise DataValidationError(f"Target column '{target_col}' not in dataframe")
            
            task_type = task_type.lower().strip()
            
            if task_type not in MODEL_RECOMMENDATIONS:
                valid_types = ", ".join(MODEL_RECOMMENDATIONS.keys())
                raise DataValidationError(
                    f"Invalid task type: {task_type}. Valid: {valid_types}"
                )
            
            logger.info(
                f"Selecting models for task: {task_type} "
                f"with {len(df)} samples, {len(df.columns)} features"
            )
            
            # Get recommendations
            recommendations = MODEL_RECOMMENDATIONS[task_type].copy()
            
            # Analyze data characteristics
            characteristics = ModelSelectionAgent._analyze_data_characteristics(
                df, target_col, task_type
            )
            
            # Adjust recommendations based on data
            recommendations = ModelSelectionAgent._adjust_recommendations(
                recommendations, characteristics, task_type
            )
            
            result = {
                "task": task_type,
                "models": recommendations["models"],
                "metrics": recommendations["metrics"],
                "default_model": recommendations.get("default_model"),
                "rationale": (
                    f"Selected {len(recommendations['models'])} models based on "
                    f"task type and data characteristics"
                ),
                "data_characteristics": characteristics,
                "hyperparameter_suggestions": (
                    ModelSelectionAgent._get_hyperparameter_suggestions(
                        task_type, characteristics
                    )
                ),
            }
            
            logger.info(
                f"Models selected: {', '.join(recommendations['models'][:3])}..."
            )
            return result
            
        except (DataValidationError, SchemaError):
            raise
        except Exception as e:
            error_msg = f"Error selecting models: {str(e)}"
            logger.error(error_msg)
            raise DataValidationError(error_msg) from e

    @staticmethod
    def _analyze_data_characteristics(
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """
        Analyze data characteristics to inform model selection.
        
        Args:
            df: Dataframe
            target_col: Target column
            task_type: Task type
            
        Returns:
            Dict: Characteristics summary
        """
        try:
            row_count = len(df)
            col_count = len(df.columns)
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            categorical_cols = len(df.select_dtypes(exclude=['number']).columns)
            
            characteristics = {
                "sample_size": row_count,
                "feature_count": col_count - 1,  # Excluding target
                "numeric_features": numeric_cols,
                "categorical_features": categorical_cols,
                "imbalance_ratio": None,
                "missing_percentage": round(
                    (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2
                ),
            }
            
            # Calculate imbalance for classification
            if task_type == "classification" and target_col in df.columns:
                value_counts = df[target_col].value_counts()
                if len(value_counts) > 0:
                    max_class = value_counts.max()
                    min_class = value_counts.min()
                    if min_class > 0:
                        characteristics["imbalance_ratio"] = round(max_class / min_class, 2)
            
            logger.debug(f"Data characteristics: {characteristics}")
            return characteristics
            
        except Exception as e:
            logger.warning(f"Error analyzing characteristics: {e}")
            return {
                "sample_size": len(df),
                "feature_count": len(df.columns) - 1,
                "error": str(e),
            }

    @staticmethod
    def _adjust_recommendations(
        recommendations: Dict,
        characteristics: Dict,
        task_type: str,
    ) -> Dict:
        """
        Adjust model recommendations based on data characteristics.
        
        Args:
            recommendations: Original recommendations
            characteristics: Data characteristics
            task_type: Task type
            
        Returns:
            Dict: Adjusted recommendations
        """
        # Large dataset - prefer scalable models
        if characteristics["sample_size"] > 100000:
            if "lightgbm" in recommendations["models"]:
                # Move LightGBM to front for large datasets
                recommendations["models"].remove("lightgbm")
                recommendations["models"].insert(0, "lightgbm")
                recommendations["default_model"] = "lightgbm"
        
        # Small dataset - prefer simpler models
        elif characteristics["sample_size"] < 1000:
            if "random_forest" in recommendations["models"]:
                recommendations["models"].remove("random_forest")
                recommendations["models"].insert(0, "random_forest")
        
        # Imbalanced classification - recommend SMOTE/class_weight
        if task_type == "classification":
            imbalance = characteristics.get("imbalance_ratio")
            if imbalance and imbalance > 3:
                logger.info(f"Imbalanced dataset detected (ratio: {imbalance})")
        
        return recommendations

    @staticmethod
    def _get_hyperparameter_suggestions(
        task_type: str,
        characteristics: Dict,
    ) -> Dict[str, Any]:
        """
        Generate hyperparameter suggestions for the task.
        
        Args:
            task_type: Task type
            characteristics: Data characteristics
            
        Returns:
            Dict: Hyperparameter suggestions
        """
        suggestions = {
            "tree_based": {
                "n_estimators": min(100 + (characteristics["sample_size"] // 1000), 500),
                "max_depth": max(5, min(15, int(characteristics["feature_count"] / 2))),
                "learning_rate": 0.1 if characteristics["sample_size"] > 10000 else 0.05,
            },
            "general": {
                "random_state": 42,
                "n_jobs": -1,
            },
        }
        
        if task_type == "classification" and characteristics.get("imbalance_ratio", 1) > 2:
            suggestions["class_weight"] = "balanced"
        
        return suggestions