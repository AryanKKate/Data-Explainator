class InsightAgent:

    @staticmethod
    def generate(

        training_results,
        explainability_report,
        target

    ):

        insights = []

        impacts = (
            explainability_report[
                "feature_impact"
            ]
        )

        total = sum(
            impacts.values()
        )

        for feature,value in list(
            impacts.items()
        )[:5]:

            pct = round(

                (
                    value
                    /
                    total
                ) * 100,

                1
            )

            insights.append({

                "type":
                "business_driver",

                "feature":
                feature,

                "importance":
                pct,

                "message":
                f"{feature} explains {pct}% of model decisions"

            })

        best_model = training_results[
            "best_model"
        ]

        result = training_results[
            "results"
        ][best_model]

        r2 = result.get(
            "r2"
        )

        if r2:

            if r2 > 0.8:

                quality = (
                    "strong"
                )

            elif r2 > 0.6:

                quality = (
                    "moderate"
                )

            else:

                quality = (
                    "weak"
                )

            insights.append({

                "type":
                "model_quality",

                "message":
                f"Model has {quality} predictive power"

            })

        return {

            "top_drivers":
            list(
                impacts.keys()
            )[:5],

            "insights":
            insights

        }