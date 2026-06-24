from analyzers.insight_ranker import InsightRanker
from analyzers.insight_narrator import InsightNarrator
class InsightAgent:



    @staticmethod
    def generate(

        training_results,
        explainability_report,
        business_intelligence,
        target

    ):
        segment_container = business_intelligence.get(
            "segments",
            {}
        )

        best_segment = segment_container.get(
            "best_segment"
        )

        insights = []



        # positive_factors = explainability_report.get(
        #     "most_positive_factors",
        #     []
        # )

        # negative_factors = explainability_report.get(
        #     "most_negative_factors",
        #     []
        # )


        observations = (

            business_intelligence.get(
                "business_observations",
                []
            )

        )

        for obs in observations:

            if obs["type"] == "relationship":

                insights.append({

                    "type":
                    "relationship",

                    "headline":
                    obs["feature"],

                    "evidence":
                    (
                        f"{obs['relationship']} association "
                        f"(correlation = {obs['correlation']})"
                    ),

                    "confidence":
                    abs(obs["correlation"]),

                    "score":
                    obs["score"]

                })

            elif obs["type"] == "driver":

                insights.append({

                    "type":
                    "driver",

                    "headline":
                    obs["feature"],

                    "evidence":
                    (
                        f"Explains "
                        f"{round(obs['importance'],2)}% "
                        f"of model behavior"
                    ),

                    "confidence":
                    round(
                        obs["importance"] / 100,
                        2
                    ),

                    "score":
                    obs["score"]

                })

            elif obs["type"] == "segment":

                insights.append({

                    "type":
                    "segment",

                    "headline":
                    obs["segment_name"],

                    "evidence":
                    (
                        f"{obs['uplift_pct']}% "
                        f"above average"
                    ),

                    "confidence":
                    0.9,

                    "score":
                    obs["score"]

                })

            elif obs["type"] == "trust":

                insights.append({

                    "type":
                    "trust",

                    "headline":
                    "Prediction Trust Score",

                    "evidence":
                    f"{obs['trust_score']}/100",

                    "confidence":
                    round(
                        obs["trust_score"] / 100,
                        2
                    ),

                    "score":
                    obs["score"]

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



        ranked = InsightRanker.rank(
            insights
        )

        for rank, item in enumerate(

            ranked,

            start=1

        ):

            item["rank"] = rank




        narratives = InsightNarrator.narrate(
            ranked
        )

        return {

            "top_insights": ranked[:10],

            "narrated_insights": narratives,

            "business_observations": observations,

            "insights": ranked

        }