import shap
import numpy as np


class ExplainabilityAgent:

    @staticmethod
    def generate(training_results):

        best_model = training_results["best_model"]

        model = (
            training_results["trained_models"]
            [best_model]
        )

        X_test = (
            training_results["X_test"]
        )

        feature_names = (
            training_results["feature_names"]
        )

        try:

            sample_size = min(
                1000,
                len(X_test)
            )

            X_sample = (
                X_test.iloc[:sample_size]
            )

            explainer = shap.Explainer(
                model,
                X_sample
            )

            shap_result = explainer(
                X_sample,
                check_additivity=False
            )

            values = shap_result.values

            # =========================
            # REGRESSION
            # (n_samples,n_features)
            # =========================

            if len(values.shape) == 2:

                shap_matrix = values

            # =========================
            # CLASSIFICATION
            # (n_samples,n_features,n_classes)
            # =========================

            elif len(values.shape) == 3:

                shap_matrix = np.mean(
                    np.abs(values),
                    axis=2
                )

            else:

                raise ValueError(
                    f"Unsupported SHAP shape: {values.shape}"
                )

            # =========================
            # GLOBAL IMPORTANCE
            # =========================

            importance = np.mean(
                np.abs(shap_matrix),
                axis=0
            )

            feature_impact = dict(

                sorted(

                    zip(
                        feature_names,
                        importance
                    ),

                    key=lambda x: x[1],

                    reverse=True

                )[:10]

            )

            total_importance = sum(
                feature_impact.values()
            )

            if total_importance > 0:

                global_importance_percentages = {

                    feature:

                    round(
                        float(score)
                        /
                        total_importance
                        * 100,
                        2
                    )

                    for feature, score
                    in feature_impact.items()

                }

            else:

                global_importance_percentages = {}

            top_drivers = list(
                feature_impact.keys()
            )[:5]

            driver_insights = [

                f"{feature} strongly influences predictions"

                for feature in top_drivers

            ]

            # =========================
            # LOCAL EXPLANATION
            # =========================

            sample_idx = 0

            local_values = (
                shap_matrix[sample_idx]
            )

            contributions = dict(

                zip(
                    feature_names,
                    local_values
                )

            )

            sample_explanation = sorted(

                contributions.items(),

                key=lambda x:
                abs(x[1]),

                reverse=True

            )[:5]

            # =========================
            # POSITIVE / NEGATIVE
            # =========================

            if len(values.shape) == 3:

                local_raw = np.mean(
                    values[sample_idx],
                    axis=1
                )

            else:

                local_raw = (
                    values[sample_idx]
                )

            raw_sample = dict(

                zip(
                    feature_names,
                    local_raw
                )

            )

            positive_factors = sorted(

                raw_sample.items(),

                key=lambda x: x[1],

                reverse=True

            )[:5]

            negative_factors = sorted(

                raw_sample.items(),

                key=lambda x: x[1]

            )[:5]

            return {

                "feature_impact":
                {
                    k: round(
                        float(v),
                        4
                    )
                    for k, v in
                    feature_impact.items()
                },

                "top_drivers":
                top_drivers,

                "driver_insights":
                driver_insights,

                "sample_explanation":
                sample_explanation,

                "global_importance_percentages":
                global_importance_percentages,

                "most_positive_factors":
                [
                    f
                    for f, _
                    in positive_factors
                ],

                "most_negative_factors":
                [
                    f
                    for f, _
                    in negative_factors
                ]

            }

        except Exception as e:

            print(
                f"SHAP failed: {e}"
            )

            return {

                "feature_impact": {},

                "top_drivers": [],

                "driver_insights": [],

                "sample_explanation": [],

                "global_importance_percentages": {},

                "most_positive_factors": [],

                "most_negative_factors": []

            }