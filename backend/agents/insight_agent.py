class InsightAgent:

    @staticmethod
    def generate(

        training_results,
        explainability_report,
        target

    ):

        insights = []

        importance = (

            explainability_report.get(
                "global_importance_percentages",
                {}
            )

        )

        # ==========================
        # Business Drivers
        # ==========================

        for feature,pct in list(

            importance.items()

        )[:5]:

            insights.append({

                "type":
                "business_driver",

                "feature":
                feature,

                "importance":
                pct,

                "message":
                f"{feature} explains {pct}% of model decisions"

            })

        # ==========================
        # Model Metrics
        # ==========================

        best_model = training_results[
            "best_model"
        ]

        result = training_results[
            "results"
        ][best_model]

        # ==========================
        # Regression Insights
        # ==========================

        if "r2" in result:

            r2 = result["r2"]

            if r2 >= 0.80:

                quality = "strong"

            elif r2 >= 0.60:

                quality = "moderate"

            else:

                quality = "weak"

            insights.append({

                "type":
                "model_quality",

                "message":
                f"Model has {quality} predictive power"

            })

            insights.append({

                "type":
                "error",

                "message":
                f"Average prediction error is {result['mae']:,.0f}"

            })

            insights.append({

                "type":
                "risk",

                "message":
                f"Worst prediction error observed was {result['max_error']:,.0f}"

            })

        # ==========================
        # Classification Insights
        # ==========================

        elif "accuracy" in result:

            accuracy = result["accuracy"]

            insights.append({

                "type":
                "model_quality",

                "message":
                f"Classification accuracy is {accuracy:.1%}"

            })

            if "f1" in result:

                insights.append({

                    "type":
                    "f1",

                    "message":
                    f"F1 Score is {result['f1']:.3f}"

                })

        # ==========================
        # Explainability Insights
        # ==========================

        positive = explainability_report.get(

            "most_positive_factors",

            []

        )

        negative = explainability_report.get(

            "most_negative_factors",

            []

        )

        if positive:

            insights.append({

                "type":
                "positive_factors",

                "message":
                f"Top positive contributors: {', '.join(positive[:3])}"

            })

        if negative:

            insights.append({

                "type":
                "negative_factors",

                "message":
                f"Top negative contributors: {', '.join(negative[:3])}"

            })

        # ==========================
        # Dataset Insights
        # ==========================

        X_train = training_results[
            "X_train"
        ]

        if X_train.isnull().sum().sum() > 0:

            insights.append({

                "type":
                "dataset",

                "message":
                "Dataset still contains missing values."

            })

        # ==========================
        # Final Output
        # ==========================

        return {

            "top_drivers":
            list(
                importance.keys()
            )[:5],

            "insights":
            insights

        }