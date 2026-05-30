class InsightAgent:

    @staticmethod
    def generate(
        training_results,
        profile,
        target
    ):

        best_model = (
            training_results["best_model"]
        )

        result = (
            training_results["results"]
            [best_model]
        )

        insights = []

        if "top_features" in result:

            features = list(
                result["top_features"].keys()
            )[:3]

            for feature in features:

                insights.append(
                    f"{feature} is a major driver of {target}"
                )

        return {

            "top_drivers":
            features,

            "insights":
            insights

        }