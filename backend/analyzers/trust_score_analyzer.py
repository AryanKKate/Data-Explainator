import numpy as np


class TrustScoreAnalyzer:

    @staticmethod
    def analyze(training_results):

        best_model = training_results[
            "best_model"
        ]

        metrics = training_results[
            "results"
        ][best_model]

        task = training_results.get(
            "task",
            "regression"
        )

        scores = {}

        # ====================================
        # PERFORMANCE SCORE
        # ====================================

        if task == "regression":

            performance_score = max(
                0,
                min(
                    metrics.get(
                        "r2",
                        0
                    ) * 100,
                    100
                )
            )

        else:

            performance_score = max(
                0,
                min(
                    metrics.get(
                        "f1",
                        0
                    ) * 100,
                    100
                )
            )

        scores[
            "performance"
        ] = round(
            performance_score,
            1
        )

        # ====================================
        # GENERALIZATION SCORE
        # ====================================

        if task == "regression":

            train_score = metrics.get(
                "r2",
                0
            )

            cv_score = metrics.get(
                "cv_r2_mean",
                train_score
            )

        else:

            train_score = metrics.get(
                "f1",
                0
            )

            cv_score = metrics.get(
                "cv_f1_mean",
                train_score
            )

        gap = abs(
            train_score - cv_score
        )

        generalization_score = max(
            0,
            100 - (gap * 300)
        )

        scores[
            "generalization"
        ] = round(
            generalization_score,
            1
        )

        # ====================================
        # STABILITY SCORE
        # ====================================

        if task == "regression":

            cv_std = metrics.get(
                "cv_r2_std",
                0.1
            )

        else:

            cv_std = metrics.get(
                "cv_f1_std",
                0.1
            )

        stability_score = max(
            0,
            100 - (cv_std * 500)
        )

        scores[
            "stability"
        ] = round(
            stability_score,
            1
        )

        # ====================================
        # DATA SUFFICIENCY
        # ====================================

        sample_count = len(
            training_results[
                "X_train"
            ]
        )

        if sample_count >= 10000:

            data_score = 100

        elif sample_count >= 5000:

            data_score = 90

        elif sample_count >= 1000:

            data_score = 80

        elif sample_count >= 500:

            data_score = 70

        elif sample_count >= 100:

            data_score = 60

        else:

            data_score = 40

        scores[
            "data_sufficiency"
        ] = data_score

        # ====================================
        # EXPLAINABILITY SCORE
        # ====================================

        model_name = best_model.lower()

        explainability_score = 100

        if "xgboost" in model_name:
            explainability_score = 85

        elif "catboost" in model_name:
            explainability_score = 85

        elif "forest" in model_name:
            explainability_score = 90

        elif "linear" in model_name:
            explainability_score = 100

        scores[
            "explainability"
        ] = explainability_score

        # ====================================
        # FINAL TRUST SCORE
        # ====================================

        trust_score = (

            scores["performance"] * 0.35 +

            scores["generalization"] * 0.25 +

            scores["stability"] * 0.15 +

            scores["data_sufficiency"] * 0.15 +

            scores["explainability"] * 0.10

        )

        trust_score = round(
            trust_score,
            1
        )

        # ====================================
        # TRUST LEVEL
        # ====================================

        if trust_score >= 85:

            trust_level = "High"

        elif trust_score >= 70:

            trust_level = "Medium"

        else:

            trust_level = "Low"

        return {

            "trust_score":
            trust_score,

            "trust_level":
            trust_level,

            "components":
            scores

        }