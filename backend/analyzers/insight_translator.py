class InsightTranslator:

    @staticmethod
    def analyze(

        opportunities,
        explainability_report,
        statistical_findings

    ):

        translated = []

        correlations = {

            item["feature"]:
            item["correlation"]

            for item in
            statistical_findings.get(
                "correlations",
                []
            )

        }

        importance_scores = (

            explainability_report.get(
                "global_importance_percentages",
                {}
            )

        )

        for item in opportunities:

            feature = item.get(
                "feature"
            )

            correlation = correlations.get(
                feature,
                None
            )

            importance = item.get(
                "importance",
                0
            )

            priority = item.get(
                "priority_score",
                0
            )

            direction = item.get(
                "direction",
                "unknown"
            )

            # ===================================
            # Relationship description
            # ===================================

            if correlation is not None:

                if correlation > 0.1:

                    relationship = (
                        "higher values are associated with higher target values"
                    )

                elif correlation < -0.1:

                    relationship = (
                        "higher values are associated with lower target values"
                    )

                else:

                    relationship = (
                        "shows weak relationship with the target"
                    )

            else:

                relationship = (
                    "is an influential predictor"
                )

            # ===================================
            # Actionability
            # ===================================

            actionable = False

            feature_lower = str(
                feature
            ).lower()

            actionable_keywords = [

                "contract",
                "discount",
                "promotion",
                "price",
                "marketing",
                "support",
                "service",
                "billing",
                "security",
                "plan",
                "subscription"

            ]

            for keyword in actionable_keywords:

                if keyword in feature_lower:

                    actionable = True
                    break

            translated.append({

                "feature":
                feature,

                "finding":
                f"{feature} {relationship}",

                "importance":
                round(
                    importance,
                    2
                ),

                "priority_score":
                round(
                    priority,
                    2
                ),

                "direction":
                direction,

                "actionable":
                actionable

            })

        return translated