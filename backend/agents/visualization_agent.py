import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

        charts = []

        y_test = training_results["y_test"]

        predictions = training_results[
            "predictions"
        ]

        feature_importance = (
            explainability_report
            .get(
                "global_importance_percentages",
                {}
            )
        )

        try:

            plt.figure(figsize=(8,6))

            plt.scatter(
                y_test,
                predictions
            )

            plt.xlabel("Actual")

            plt.ylabel("Predicted")

            plt.title(
                "Actual vs Predicted"
            )

            path = (
                "outputs/charts/"
                "actual_vs_predicted.png"
            )

            plt.savefig(path)

            plt.close()

            charts.append(path)

        except:
            pass

        try:

            residuals = (
                y_test -
                predictions
            )

            plt.figure(figsize=(8,6))

            sns.histplot(
                residuals,
                kde=True
            )

            plt.title(
                "Residual Distribution"
            )

            path = (
                "outputs/charts/"
                "residual_distribution.png"
            )

            plt.savefig(path)

            plt.close()

            charts.append(path)

        except:
            pass

        try:

            plt.figure(figsize=(10,6))

            features = list(
                feature_importance.keys()
            )[:10]

            values = list(
                feature_importance.values()
            )[:10]

            plt.barh(
                features,
                values
            )

            plt.title(
                "Feature Importance"
            )

            path = (
                "outputs/charts/"
                "feature_importance.png"
            )

            plt.savefig(path)

            plt.close()

            charts.append(path)

        except:
            pass

        return {

            "charts":
            charts
        }