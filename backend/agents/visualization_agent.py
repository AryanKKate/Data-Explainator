import os
import matplotlib.pyplot as plt
import numpy as np


class VisualizationAgent:

    @staticmethod
    def generate(

        training_results,
        explainability_report

    ):

        os.makedirs(
            "outputs/charts",
            exist_ok=True
        )

        charts = {}

        best_model = training_results[
            "best_model"
        ]

        y_test = training_results[
            "y_test"
        ]

        X_test = training_results[
            "X_test"
        ]

        model = training_results[
            "trained_models"
        ][best_model]

        predictions = model.predict(
            X_test
        )

        # =====================================
        # Actual vs Predicted
        # =====================================

        try:

            plt.figure(
                figsize=(8,6)
            )

            plt.scatter(
                y_test,
                predictions
            )

            plt.xlabel(
                "Actual Values"
            )

            plt.ylabel(
                "Predicted Values"
            )

            plt.title(
                "Actual vs Predicted"
            )

            path = (
                "outputs/charts/"
                "actual_vs_predicted.png"
            )

            plt.savefig(
                path,
                bbox_inches="tight"
            )

            plt.close()

            charts[
                "actual_vs_predicted"
            ] = path

        except Exception as e:

            print(
                f"Actual vs Predicted failed: {e}"
            )

        # =====================================
        # Residual Plot
        # =====================================

        try:

            residuals = (

                y_test -
                predictions

            )

            plt.figure(
                figsize=(8,6)
            )

            plt.scatter(

                predictions,

                residuals

            )

            plt.axhline(
                y=0,
                linestyle="--"
            )

            plt.xlabel(
                "Predicted"
            )

            plt.ylabel(
                "Residual"
            )

            plt.title(
                "Residual Plot"
            )

            path = (
                "outputs/charts/"
                "residual_plot.png"
            )

            plt.savefig(
                path,
                bbox_inches="tight"
            )

            plt.close()

            charts[
                "residual_plot"
            ] = path

        except Exception as e:

            print(
                f"Residual plot failed: {e}"
            )

        # =====================================
        # Error Distribution
        # =====================================

        try:

            errors = np.abs(

                y_test -
                predictions

            )

            plt.figure(
                figsize=(8,6)
            )

            plt.hist(
                errors,
                bins=30
            )

            plt.xlabel(
                "Prediction Error"
            )

            plt.ylabel(
                "Frequency"
            )

            plt.title(
                "Error Distribution"
            )

            path = (
                "outputs/charts/"
                "error_distribution.png"
            )

            plt.savefig(
                path,
                bbox_inches="tight"
            )

            plt.close()

            charts[
                "error_distribution"
            ] = path

        except Exception as e:

            print(
                f"Error distribution failed: {e}"
            )

        # =====================================
        # Feature Importance
        # =====================================

        try:

            impact = (

                explainability_report[
                    "global_importance_percentages"
                ]

            )

            features = list(
                impact.keys()
            )[:10]

            values = list(
                impact.values()
            )[:10]

            plt.figure(
                figsize=(10,6)
            )

            plt.barh(
                features,
                values
            )

            plt.title(
                "Feature Importance (%)"
            )

            plt.xlabel(
                "Importance %"
            )

            plt.gca().invert_yaxis()

            path = (
                "outputs/charts/"
                "feature_importance.png"
            )

            plt.savefig(
                path,
                bbox_inches="tight"
            )

            plt.close()

            charts[
                "feature_importance"
            ] = path

        except Exception as e:

            print(
                f"Feature importance failed: {e}"
            )

        return {

            "charts":
            charts

        }