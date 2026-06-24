class ObservationGenerator:

    @staticmethod
    def analyze(
        target,
        stats,
        segments,
        explainability_report,
        trust_score=None
    ):

        observations = []

        # =====================================
        # Segment Observations
        # =====================================

        best_segment = segments.get(
            "best_segment",
            None
        )

        if best_segment:

            uplift = round(
                best_segment["uplift_pct"],
                2
            )

            observations.append({

                "type":"segment",

                "segment_name":
                best_segment["segment_name"],

                "uplift_pct":
                uplift,

                "records":
                best_segment["records"],

                "percentage":
                best_segment["percentage"],

                "score":
                abs(uplift)

            })

        # =====================================
        # Relationship Observations
        # =====================================

        drivers = stats.get(
            "driver_analysis",
            []
        )

        for item in drivers[:5]:

            corr = abs(
                item["correlation"]
            )

            if corr < 0.3:
                continue

                observations.append({

                    "type":"relationship",

                    "feature":
                    item["feature"],

                    "relationship":
                    item["relationship"],

                    "strength":
                    item["strength"],

                    "correlation":
                    item["correlation"],

                    "score":
                    abs(corr) * 100

                })

        # =====================================
        # Driver Observations
        # =====================================

        importance = (

            explainability_report.get(
                "global_importance_percentages",
                {}
            )

        )

        top_drivers = (

            explainability_report.get(
                "top_drivers",
                []
            )

        )

        for feature in top_drivers[:3]:

            score = importance.get(
                feature,
                0
            )

            observations.append({

                "type":"driver",

                "feature":
                feature,

                "importance":
                score,

                "score":
                score

            })

        # =====================================
        # Trust Observation
        # =====================================

        if trust_score is not None:

            observations.append({

                "type":"trust",

                "trust_score":
                trust_score,

                "score":
                trust_score * 0.2

            })

        return sorted(

            observations,

            key=lambda x:
            x["score"],

            reverse=True

        )