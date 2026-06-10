class OpportunityAnalyzer:

    @staticmethod
    def analyze(

        explainability_report,
        statistical_findings,
        trust

    ):
        trust_score = trust['trust_score']
        opportunities = []

        shap_scores = (

            explainability_report
            .get(
                "global_importance_percentages",
                {}
            )

        )

        drivers = statistical_findings[
            "driver_analysis"
        ]

        for item in drivers:

            feature = item["feature"]

            uplift = abs(
                item["uplift_pct"]
            )

            p_value = item["p_value"]

            importance = shap_scores.get(
                feature,
                0
            )

            significance = max(

                0,

                1 - p_value

            )

            score = (

                uplift

                *

                importance

                *

                significance

                *

                (
                    trust_score
                    /
                    100
                )

            )

            opportunities.append({

                "feature":
                feature,

                "uplift_pct":
                uplift,

                "importance":
                importance,

                "p_value":
                p_value,

                "priority_score":
                round(
                    score,
                    2
                )

            })

        return sorted(

            opportunities,

            key=lambda x:
            x[
                "priority_score"
            ],

            reverse=True

        )[:10]