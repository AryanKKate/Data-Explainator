class InsightAgent:

    @staticmethod
    def generate(
        training_results,
        profile,
        target,
        explainability_report=None
    ):

        best_model = training_results["best_model"]

        result = training_results["results"][best_model]

        insights = []

        top_drivers = []

        # -------------------------
        # Feature based insights
        # -------------------------

        if "top_features" in result:

            top_drivers = list(
                result["top_features"].keys()
            )[:5]

            for feature in top_drivers:

                insights.append({

                    "type":
                    "feature_driver",

                    "feature":
                    feature,

                    "message":
                    f"{feature} strongly influences {target}"

                })

        # -------------------------
        # Model quality insights
        # -------------------------

        if "r2" in result:

            r2 = result["r2"]

            if r2 >= 0.8:

                insights.append({

                    "type":
                    "model_quality",

                    "message":
                    "Model explains most variation in the target."

                })

            elif r2 >= 0.6:

                insights.append({

                    "type":
                    "model_quality",

                    "message":
                    "Model has moderate predictive power."

                })

            else:

                insights.append({

                    "type":
                    "model_quality",

                    "message":
                    "Model accuracy is limited. Additional features may be required."

                })

        # -------------------------
        # Error analysis
        # -------------------------

        if "mean_error" in result:

            insights.append({

                "type":
                "error",

                "message":
                f"Average prediction error is {result['mean_error']:,.0f}"

            })

        if "max_error" in result:

            insights.append({

                "type":
                "risk",

                "message":
                f"Worst prediction error observed was {result['max_error']:,.0f}"

            })

        # -------------------------
        # Dataset observations
        # -------------------------

        if profile:

            insights.append({

                "type":
                "dataset",

                "message":
                f"Dataset contains {profile.get('rows', 'unknown')} records."

            })

        return {

            "top_drivers":
            top_drivers,

            "insights":
            insights

        }