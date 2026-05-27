class ModelSelectionAgent:

    @staticmethod
    def select(
        intent,
        schema,
        df,
        target
    ):

        rows = len(df)
        cols = len(df.columns)

        result = {}

        # Classification
        if intent == "classification":

            result = {

                "task":"classification",

                "models":[
                    "xgboost",
                    "lightgbm",
                    "catboost",
                    "random_forest"
                ],

                "metrics":[
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc"
                ]
            }

        # Regression
        elif intent == "regression":

            result = {

                "task":"regression",

                "models":[
                    "xgboost",
                    "lightgbm",
                    "catboost",
                    "random_forest"
                ],

                "metrics":[
                    "rmse",
                    "mae",
                    "r2"
                ]
            }

        elif intent=="clustering":

            result={

                "task":"clustering",

                "models":[
                    "kmeans",
                    "dbscan"
                ],

                "metrics":[
                    "silhouette"
                ]
            }

        return result