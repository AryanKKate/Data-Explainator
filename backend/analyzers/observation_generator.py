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

                "type":
                "segment",

                "observation":
                (
                    f"A segment representing "
                    f"{best_segment['percentage']}% "
                    f"of records achieves "
                    f"{uplift}% higher "
                    f"{target} than average."
                ),

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

                "type":
                "relationship",

                "observation":
                (
                    f"{item['feature']} "
                    f"shows a "
                    f"{item['strength']} "
                    f"{item['relationship']} "
                    f"association with "
                    f"{target}."
                ),

                "score":
                round(
                    corr * 100,
                    2
                )

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

                "type":
                "driver",

                "observation":
                (
                    f"{feature} is among "
                    f"the strongest drivers "
                    f"of model predictions."
                ),

                "score":
                round(score, 2)

            })

        # =====================================
        # Trust Observation
        # =====================================

        if trust_score is not None:

            observations.append({

                "type":
                "trust",

                "observation":
                (
                    f"Prediction trust score "
                    f"is {trust_score}/100."
                ),

                "score":
                trust_score * 0.2

            })

        return sorted(

            observations,

            key=lambda x:
            x["score"],

            reverse=True

        )