class InsightAgent:

    @staticmethod
    def generate(

        training_results,
        explainability_report,
        business_intelligence,
        target

    ):

        insights = []

        # =====================================
        # SHAP Drivers
        # =====================================

        importance = (

            explainability_report.get(
                "global_importance_percentages",
                {}
            )

        )

        for feature, pct in list(
            importance.items()
        )[:5]:

            insights.append({

                "type":
                "driver",

                "headline":
                f"{feature} is a key driver",

                "evidence":
                f"Explains {pct}% of model behavior",

                "confidence":
                round(
                    pct / 100,
                    2
                )

            })

        # =====================================
        # Statistical Opportunities
        # =====================================

        statistical_findings = (

            business_intelligence
            .get(
                "statistical_findings",
                {}
            )
            .get(
                "driver_analysis",
                []
            )

        )

        for item in statistical_findings[:5]:

            insights.append({

                "type":
                "opportunity",

                "headline":
                f"{item['feature']} creates value",

                "evidence":
                f"{item['uplift_pct']}% uplift",

                "confidence":
                round(
                    max(
                        0.5,
                        1 - item["p_value"]
                    ),
                    2
                )

            })

        # =====================================
        # Best Segment
        # =====================================

        segments = (

            business_intelligence
            .get(
                "segments",
                {}
            )
            .get(
                "segments",
                []
            )

        )

        if len(segments) > 0:

            best_segment = segments[0]

            insights.append({

                "type":
                "segment",

                "headline":
                f"{best_segment['segment']} is highest value",

                "evidence":
                f"{best_segment['uplift_pct']}% above average",

                "confidence":
                0.9

            })

        # =====================================
        # Data Quality
        # =====================================

        quality = business_intelligence.get(
            "data_quality",
            {}
        )

        quality_score = quality.get(
            "overall_score",
            None
        )

        if quality_score is not None:

            insights.append({

                "type":
                "quality",

                "headline":
                "Data Quality Score",

                "evidence":
                f"{quality_score}/100",

                "confidence":
                round(
                    quality_score / 100,
                    2
                )

            })

        # =====================================
        # Model Reliability
        # =====================================

        best_model = training_results[
            "best_model"
        ]

        result = training_results[
            "results"
        ][best_model]

        if "r2" in result:

            reliability = result["r2"]

            metric_name = "R²"

        else:

            reliability = result.get(
                "f1",
                0
            )

            metric_name = "F1"

        insights.append({

            "type":
            "model",

            "headline":
            "Model Reliability",

            "evidence":
            f"{metric_name} = {round(reliability,3)}",

            "confidence":
            round(
                reliability,
                2
            )

        })

        # =====================================
        # Trust Score
        # =====================================

        trust = business_intelligence.get(
            "trust_score",
            {}
        )

        if isinstance(trust, dict):

            insights.append({

                "type":
                "trust",

                "headline":
                "Prediction Trust Score",

                "evidence":
                f"{trust.get('trust_score',0)}/100",

                "confidence":
                round(
                    trust.get(
                        "trust_score",
                        0
                    ) / 100,
                    2
                )

            })

        return {

            "insights":
            insights

        }