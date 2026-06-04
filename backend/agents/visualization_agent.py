import matplotlib.pyplot as plt
import seaborn as sns
import os


class VisualizationAgent:

    @staticmethod
    def generate(training_results):

        os.makedirs(
            "outputs/charts",
            exist_ok=True
        )

        best_model = training_results[
            "best_model"
        ]

        y_test = training_results[
            "y_test"
        ]

        best_model_name = training_results[
            "best_model"
        ]

        model = training_results[
            "trained_models"
        ][best_model_name]

        X_test = training_results[
            "X_test"
        ]

        predictions = model.predict(
            X_test
        )

        charts = []

        try:

            plt.figure(figsize=(8,6))

            plt.scatter(
                y_test,
                predictions
            )

            plt.xlabel(
                "Actual"
            )

            plt.ylabel(
                "Predicted"
            )

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

        return charts