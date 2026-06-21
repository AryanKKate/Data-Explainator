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

            importance = importance_scores.get(
                feature,
                item.get(
                    "importance",
                    0
                )
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
            # Relationship
            # ===================================

            if correlation is not None:

                if correlation > 0:

                    relationship = "positive"

                elif correlation < 0:

                    relationship = "negative"

                else:

                    relationship = "neutral"

                abs_corr = abs(
                    correlation
                )

                if abs_corr >= 0.60:

                    strength = "strong"

                elif abs_corr >= 0.30:

                    strength = "moderate"

                else:

                    strength = "weak"

            else:

                relationship = "unknown"

                strength = "unknown"

            # ===================================
            # Business Finding
            # ===================================

            if correlation is not None:

                finding = (

                    f"{feature} explains "
                    f"{round(importance,2)}% "
                    f"of model behavior and "
                    f"shows a {strength} "
                    f"{relationship} relationship "
                    f"with the target"

                )

            else:

                finding = (

                    f"{feature} explains "
                    f"{round(importance,2)}% "
                    f"of model behavior"

                )

            # ===================================
            # Dataset-Agnostic
            # ===================================

            translated.append({

                "feature":
                feature,

                "finding":
                finding,

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

                "correlation":
                correlation,

                "relationship":
                relationship,

                "strength":
                strength,

                "actionable":
                False

            })

        return sorted(

            translated,

            key=lambda x:
            x["priority_score"],

            reverse=True

        )