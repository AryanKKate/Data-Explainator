import shap
import numpy as np


class ExplainabilityAgent:

    @staticmethod
    def generate(training_results):

        best_model = training_results[
            "best_model"
        ]

        model = training_results[
            "trained_models"
        ][best_model]

        X_test = training_results[
            "X_test"
        ]

        feature_names = training_results[
            "feature_names"
        ]

        try:

            explainer = shap.Explainer(
                model,
                X_test
            )

            shap_values = explainer(
                X_test
            )

            importance = np.abs(
                shap_values.values
            ).mean(axis=0)

            feature_impact = dict(

                sorted(

                    zip(
                        feature_names,
                        importance
                    ),

                    key=lambda x: x[1],

                    reverse=True

                )

            )

            feature_impact = dict(

                list(
                    feature_impact.items()
                )[:10]

            )

            driver_insights = [

                f"{feature} strongly influences predictions"

                for feature in list(
                    feature_impact.keys()
                )[:5]

            ]

            sample_idx = 0

            contributions = dict(

                zip(

                    feature_names,

                    shap_values.values[
                        sample_idx
                    ]

                )

            )

            sample_explanation = sorted(

                contributions.items(),

                key=lambda x: abs(x[1]),

                reverse=True

            )[:5]

            return {

                "feature_impact":
                feature_impact,

                "driver_insights":
                driver_insights,

                "sample_explanation":
                sample_explanation

            }

        except Exception as e:

            print(
                f"SHAP failed: {e}"
            )

            return {

                "feature_impact": {},

                "driver_insights": [],

                "sample_explanation": []

            }