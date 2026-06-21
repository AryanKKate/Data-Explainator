class OpportunityAnalyzer:


    @staticmethod
    def analyze(

        explainability_report,
        statistical_findings,
        trust

    ):

        trust_score = trust["trust_score"]

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

            correlation = item[
                "correlation"
            ]

            p_value = item[
                "p_value"
            ]

            importance = (
                shap_scores.get(
                    feature,
                    0
                )
            )

            significance = max(
                0,
                1 - p_value
            )

            # keep direction

            direction = item[
                "relationship"
            ]

            # weighted score

            score = (

                abs(correlation) * 100 * 0.6

                +

                importance * 0.3

                +

                significance * 100 * 0.1

            )

            score = (
                score
                *
                (
                    trust_score
                    /
                    100
                )
            )
            effect_strength = "weak"

            if abs(correlation) >= 0.5:
                effect_strength = "strong"

            elif abs(correlation) >= 0.3:
                effect_strength = "moderate"

            elif abs(correlation) >= 0.1:
                effect_strength = "weak"

            opportunities.append({

                "feature":
                feature,

                "correlation":
                round(
                    correlation,
                    3
                ),
                "effect_strength":
                 effect_strength,

                "direction":
                direction,

                "importance":
                round(
                    importance,
                    2
                ),

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