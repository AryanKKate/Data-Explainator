"""
Training agent for model fitting and evaluation.
Handles multiple model types with comprehensive metrics.
"""

from typing import Dict, Any, Literal, Optional, Tuple
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from exceptions import ModelTrainingError, DataValidationError

logger = logging.getLogger(__name__)

TaskType = Literal["classification", "regression", "clustering"]


class TrainingAgent:
    """Trains and evaluates ML models."""

    # Default hyperparameters
    DEFAULT_PARAMS = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": 0,
        },
        "lightgbm": {
            "n_estimators": 100,
            "max_depth": 7,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": -1,
        },
        "catboost": {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": 0,
        },
    }

    @staticmethod
    def _get_model_instance(model_name: str, task_type: str) -> Any:
        """
        Get model instance by name.
        
        Args:
            model_name: Model name
            task_type: Task type
            
        Returns:
            Model instance
            
        Raises:
            ModelTrainingError: If model unavailable
        """
        try:
            params = TrainingAgent.DEFAULT_PARAMS.get(model_name, {})
            
            if model_name == "random_forest":
                if task_type == "classification":
                    return RandomForestClassifier(**params)
                else:
                    return RandomForestRegressor(**params)
            
            elif model_name == "xgboost":
                from xgboost import XGBClassifier, XGBRegressor
                if task_type == "classification":
                    return XGBClassifier(**params)
                else:
                    return XGBRegressor(**params)
            
            elif model_name == "lightgbm":
                from lightgbm import LGBMClassifier, LGBMRegressor
                if task_type == "classification":
                    return LGBMClassifier(**params)
                else:
                    return LGBMRegressor(**params)
            
            elif model_name == "catboost":
                from catboost import CatBoostClassifier, CatBoostRegressor
                if task_type == "classification":
                    return CatBoostClassifier(**params)
                else:
                    return CatBoostRegressor(**params)
            
            else:
                raise ModelTrainingError(f"Unknown model: {model_name}")
        
        except ImportError as e:
            raise ModelTrainingError(
                f"Model '{model_name}' not installed: {str(e)}"
            ) from e
        except Exception as e:
            raise ModelTrainingError(
                f"Error initializing {model_name}: {str(e)}"
            ) from e

    @staticmethod
    def _evaluate_classification(
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate classification model.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dict: Evaluation metrics
        """
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        }
        
        # ROC AUC for binary classification
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                metrics["roc_auc"] = round(roc_auc_score(y_test, y_pred_proba[:, 1]), 4)
            except Exception as e:
                logger.debug(f"Could not calculate ROC AUC: {e}")
        
        return metrics

    @staticmethod
    def _evaluate_regression(
        y_test: pd.Series,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate regression model.
        
        Args:
            y_test: True values
            y_pred: Predicted values
            
        Returns:
            Dict: Evaluation metrics
        """
        mse = mean_squared_error(y_test, y_pred)
        
        metrics = {
            "rmse": round(np.sqrt(mse), 4),
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "r2": round(r2_score(y_test, y_pred), 4),
        }
        
        # MAPE if no zero values in y_test
        if (y_test != 0).all():
            try:
                metrics["mape"] = round(mean_absolute_percentage_error(y_test, y_pred), 4)
            except Exception as e:
                logger.debug(f"Could not calculate MAPE: {e}")
        
        return metrics

    @staticmethod
    def train(
        df: pd.DataFrame,
        target: str,
        model_plan: Dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train and evaluate models.
        
        Args:
            df: Input dataframe
            target: Target column name
            model_plan: Model selection plan
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            Dict: Training results with metrics
            
        Raises:
            ModelTrainingError: If training fails
            DataValidationError: If data invalid
        """
        try:
            # Validate inputs
            if df is None or df.empty:
                raise DataValidationError("Dataframe cannot be None or empty")
            
            if target not in df.columns:
                raise DataValidationError(f"Target column '{target}' not found")
            
            if not isinstance(model_plan, dict) or "task" not in model_plan:
                raise DataValidationError("Invalid model_plan structure")
            
            logger.info(
                f"Starting training for task: {model_plan['task']} "
                f"with {len(df)} samples"
            )
            
            # Prepare data
            y = df[target]
            X = df.drop(columns=[target])
            
            # Check for missing values
            if X.isnull().any().any() or y.isnull().any():
                logger.warning("Missing values detected in data - ensure cleaning is complete")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if model_plan["task"] == "classification" else None
            )
            
            logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
            
            task_type = model_plan["task"]
            results = {}
            best_model = None
            best_score = -float("inf")
            
            # Train each model
            for model_name in model_plan.get("models", []):
                try:
                    logger.info(f"Training {model_name}...")
                    
                    # Get model
                    model = TrainingAgent._get_model_instance(model_name, task_type)
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    y_pred_proba = None
                    
                    if task_type == "classification":
                        try:
                            y_pred_proba = model.predict_proba(X_test)
                        except AttributeError:
                            logger.debug(f"{model_name} doesn't support predict_proba")
                        
                        metrics = TrainingAgent._evaluate_classification(
                            y_test, y_pred, y_pred_proba
                        )
                        primary_metric = metrics.get("f1", metrics.get("accuracy", 0))
                    
                    else:  # Regression
                        metrics = TrainingAgent._evaluate_regression(y_test, y_pred)
                        primary_metric = metrics.get("r2", 0)
                    
                    # Store results
                    results[model_name] = {
                        "metrics": metrics,
                        "primary_metric": primary_metric,
                        "status": "success",
                    }
                    
                    # Track best model
                    if primary_metric > best_score:
                        best_score = primary_metric
                        best_model = model_name
                    
                    logger.info(f"{model_name} trained. Primary metric: {primary_metric}")
                
                except Exception as e:
                    error_msg = f"Error training {model_name}: {str(e)}"
                    logger.error(error_msg)
                    results[model_name] = {
                        "status": "failed",
                        "error": str(e),
                    }
            
            # Prepare final results
            final_results = {
                "task": task_type,
                "models_trained": list(results.keys()),
                "results": results,
                "best_model": best_model,
                "best_score": round(best_score, 4),
                "test_size": test_size,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
            
            if not best_model:
                logger.warning("No models trained successfully")
            else:
                logger.info(f"Training complete. Best model: {best_model}")
            
            return final_results
        
        except (DataValidationError, ModelTrainingError):
            raise
        except Exception as e:
            error_msg = f"Unexpected error during training: {str(e)}"
            logger.error(error_msg)
            raise ModelTrainingError(error_msg) from e
