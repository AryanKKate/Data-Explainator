from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import numpy as np


class TrainingAgent:

    @staticmethod
    def train(
        df,
        target,
        model_plan
    ):

        y = df[target]

        X = df.drop(
            columns=[target]
        )

        X_train, X_test, y_train, y_test = (

            train_test_split(

                X,
                y,
                test_size=0.2,
                random_state=42

            )

        )

        task = model_plan["task"]

        results = {}

        best_model = None
        best_score = -999

        available_models = {}

        # Classification models

        if task=="classification":

            available_models={

                "random_forest":
                RandomForestClassifier(
                    random_state=42
                )

            }

            try:

                from xgboost import XGBClassifier

                available_models[
                    "xgboost"
                ]=XGBClassifier(
                    random_state=42
                )

            except:
                pass


            try:

                from lightgbm import LGBMClassifier

                available_models[
                    "lightgbm"
                ]=LGBMClassifier(
                    random_state=42
                )

            except:
                pass


            try:

                from catboost import CatBoostClassifier

                available_models[
                    "catboost"
                ]=(

                    CatBoostClassifier(
                        verbose=0,
                        random_state=42
                    )

                )

            except:
                pass


        # Regression models

        elif task=="regression":

            available_models={

                "random_forest":
                RandomForestRegressor(
                    random_state=42
                )

            }

        for model_name in model_plan["models"]:

            if model_name not in available_models:

                continue

            model=available_models[
                model_name
            ]

            print(
                f"\nTraining {model_name}"
            )

            model.fit(
                X_train,
                y_train
            )

            pred=model.predict(
                X_test
            )

            if task=="classification":

                accuracy=accuracy_score(
                    y_test,
                    pred
                )

                precision=precision_score(
                    y_test,
                    pred
                )

                recall=recall_score(
                    y_test,
                    pred
                )

                f1=f1_score(
                    y_test,
                    pred
                )

                results[
                    model_name
                ]={

                    "accuracy":
                    round(
                        accuracy,
                        3
                    ),

                    "precision":
                    round(
                        precision,
                        3
                    ),

                    "recall":
                    round(
                        recall,
                        3
                    ),

                    "f1":
                    round(
                        f1,
                        3
                    )
                }

                if f1>best_score:

                    best_score=f1

                    best_model=model_name

            elif task=="regression":

                rmse=np.sqrt(
                    mean_squared_error(
                        y_test,
                        pred
                    )
                )

                mae=mean_absolute_error(
                    y_test,
                    pred
                )

                r2=r2_score(
                    y_test,
                    pred
                )

                results[
                    model_name
                ]={

                    "rmse":
                    round(
                        rmse,
                        3
                    ),

                    "mae":
                    round(
                        mae,
                        3
                    ),

                    "r2":
                    round(
                        r2,
                        3
                    )

                }

                if r2>best_score:

                    best_score=r2

                    best_model=model_name


        return {

            "best_model":
            best_model,

            "results":
            results
        }