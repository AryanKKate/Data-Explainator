class ModelSelectionAgent:

    @staticmethod
    def select(
        intent,
        schema,
        df,
        target
    ):

        rows=len(df)

        cols=len(df.columns)

        feature_count=cols-1

        numeric_features=len(
            df.select_dtypes(
                include=["number"]
            ).columns
        )

        categorical_features=len(
            df.select_dtypes(
                exclude=["number"]
            ).columns
        )

        missing_ratio=(

            df.isnull().sum().sum()

            /

            (rows*cols)

        )

        high_dimensional=feature_count>100

        small_dataset=rows<1000

        large_dataset=rows>100000

        sparse_dataset=missing_ratio>0.3

        imbalanced=False

        if intent=="classification":

            imbalance_ratio=(

                df[target]
                .value_counts(normalize=True)
                .min()

            )

            imbalanced=imbalance_ratio<0.2

        result={

            "task":intent,

            "models":[],

            "metrics":[],

            "steps":[],

            "cross_validation":False,

            "ensemble":False

        }

        if intent=="classification":

            result["metrics"]=[

                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc"

            ]

            if small_dataset:

                result["models"]=[

                    "random_forest",
                    "logistic_regression"

                ]

                result[
                    "cross_validation"
                ]=True

            else:

                result["models"]=[

                    "xgboost",
                    "lightgbm",
                    "catboost",
                    "random_forest"

                ]

            if imbalanced:

                result["steps"].append(
                    "smote"
                )

        elif intent=="regression":

            result["metrics"]=[

                "rmse",
                "mae",
                "r2"

            ]

            if small_dataset:

                result["models"]=[

                    "random_forest",
                    "linear_regression"

                ]

                result[
                    "cross_validation"
                ]=True

            else:

                result["models"]=[

                    "xgboost",
                    "lightgbm",
                    "catboost",
                    "random_forest"

                ]

        elif intent=="clustering":

            result["models"]=[

                "kmeans",
                "dbscan"

            ]

            result["metrics"]=[

                "silhouette"

            ]

        if numeric_features>0:

            result["steps"].append(
                "scaling"
            )

        if categorical_features>0:

            result["steps"].append(
                "encoding"
            )

        if sparse_dataset:

            result["steps"].append(
                "advanced_imputation"
            )

        if high_dimensional:

            result["steps"].append(
                "feature_selection"
            )

        if len(result["models"])>=3:

            result["ensemble"]=True

        return result