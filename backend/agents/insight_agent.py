class InsightAgent:
    @staticmethod
    def narrate(insights):

        narratives = []

        for item in insights[:5]:

            if item["type"] == "relationship":

                narratives.append(

                    f"{item['headline']} shows "
                    f"{item['evidence']}."

                )

            elif item["type"] == "driver":

                narratives.append(

                    f"{item['headline']} is one "
                    f"of the strongest model drivers."

                )

            elif item["type"] == "segment":

                narratives.append(

                    f"{item['headline']} "
                    f"with {item['evidence']}."

                )

            elif item["type"] == "model":

                narratives.append(

                    f"The model achieved "
                    f"{item['evidence']}."

                )

            elif item["type"] == "quality":

                narratives.append(

                    f"Data quality remains high "
                    f"at {item['evidence']}."

                )

            elif item["type"] == "trust":

                narratives.append(

                    f"Prediction trust score is "
                    f"{item['evidence']}."

                )

        return narratives

    @staticmethod
    def rank_insights(insights):

        for insight in insights:

            score = 0

            if insight["type"] == "relationship":

                score = (
                    insight["confidence"] * 100
                )

            elif insight["type"] == "segment":

                uplift = float(
                    insight["evidence"]
                    .split("%")[0]
                )

                score = abs(uplift)

            elif insight["type"] == "driver":

                try:

                    score = float(
                        insight["evidence"]
                        .split("Explains ")[1]
                        .split("%")[0]
                    )

                except:
                    score = 0

            elif insight["type"] == "model":

                score = (
                    insight["confidence"] * 40
                )

            elif insight["type"] == "quality":

                score = (
                    insight["confidence"] * 20
                )

            elif insight["type"] == "trust":

                score = (
                    insight["confidence"] * 20
                )

            insight["score"] = round(
                score,
                2
            )

        insights = sorted(

            insights,

            key=lambda x:
            x["score"],

            reverse=True

        )

        for rank, item in enumerate(
            insights,
            start=1
        ):

            item["rank"] = rank

        return insights

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

        # positive_factors = explainability_report.get(
        #     "most_positive_factors",
        #     []
        # )

        # negative_factors = explainability_report.get(
        #     "most_negative_factors",
        #     []
        # )

        for feature, pct in list(
            importance.items()
        )[:5]:

            insights.append({

                "type":
                "driver",

                "headline":
                feature,

                "evidence":
                (
                    f"Explains {pct}% "
                    f"of model behavior"
                ),

                "confidence":
                round(
                    pct / 100,
                    2
                ),

                "score":
                float(pct)

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

            relationship = item[
                "relationship"
            ]

            corr_strength = abs(
                item["correlation"]
            )

            insights.append({

                "type":
                "relationship",

                "headline":
                item["feature"],

                "evidence":
                (
                    f"{relationship} association "
                    f"(correlation = "
                    f"{item['correlation']})"
                ),

                "confidence":
                round(
                    corr_strength,
                    2
                ),

                "score":
                round(
                    corr_strength * 100,
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

            best_segment = (
                business_intelligence
                .get(
                    "segments",
                    {}
                )
                .get(
                    "best_segment",
                    None
                )
            )

            if best_segment:

                segment_score = (

                    abs(
                        best_segment["uplift_pct"]
                    )

                    +

                    best_segment["percentage"] * 0.3

                )

                insights.append({

                    "type":
                    "segment",

                    "headline":
                    (
                        f"High-value segment: "
                        f"{best_segment['segment_name']}"
                    ),

                    "evidence":
                    (
                        f"{best_segment['uplift_pct']}% "
                        f"above average"
                    ),

                    "confidence":
                    0.9,

                    "score":
                    round(
                        segment_score,
                        2
                    )

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
                ),

                "score":
                float(
                    quality_score
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
            ),

            "score":
            round(
                reliability * 100,
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

            trust_score = trust.get(
            "trust_score",
            0
        )

            insights.append({

                "type":
                "trust",

                "headline":
                "Prediction Trust Score",

                "evidence":
                f"{trust_score}/100",

                "confidence":
                round(
                    trust_score / 100,
                    2
                ),

                "score":
                float(
                    trust_score
                )

            })

        insights = sorted(

            insights,

            key=lambda x:
            x["score"],

            reverse=True

        )

        for rank, item in enumerate(
            insights,
            start=1
        ):

            item["rank"] = rank

        ranked = InsightAgent.rank_insights(
            insights
        )

        narratives = InsightAgent.narrate(
            ranked
        )

        return {

            "top_insights":
            ranked[:10],

            "narratives":
            narratives,

            "insights":
            ranked

        }