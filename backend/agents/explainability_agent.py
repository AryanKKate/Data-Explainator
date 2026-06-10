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

            values = shap_values.values

            # Handle multiclass outputs
            if len(values.shape) == 3:

                values = np.mean(
                    np.abs(values),
                    axis=2
                )

            importance = np.abs(
                values
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


            total_importance = sum(

                feature_impact.values()

            )

            if total_importance > 0:

                global_importance_percentages = {

                    feature:

                    round(
                        score /
                        total_importance
                        * 100,
                        2
                    )

                    for feature, score in
                    feature_impact.items()

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


            sample_idx = 0

            contributions = dict(

                zip(

                    feature_names,

                    values[
                        sample_idx
                    ]

                )

            )

            sample_explanation = sorted(

                contributions.items(),

                key=lambda x: abs(x[1]),

                reverse=True

            )[:5]


            raw_sample = dict(

                zip(

                    feature_names,

                    shap_values.values[
                        sample_idx
                    ]

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

            most_positive_factors = [

                feature

                for feature, _
                in positive_factors

            ]

            most_negative_factors = [

                feature

                for feature, _
                in negative_factors

            ]

            return {

                "feature_impact":
                feature_impact,

                "top_drivers":
                top_drivers,

                "driver_insights":
                driver_insights,

                "sample_explanation":
                sample_explanation,

                "global_importance_percentages":
                global_importance_percentages,

                "most_positive_factors":
                most_positive_factors,

                "most_negative_factors":
                most_negative_factors

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