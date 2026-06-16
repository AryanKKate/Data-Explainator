import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


class StatisticalAnalyzer:

    @staticmethod
    def analyze(df, target):

        findings = []

        correlations = []

        numeric_cols = [

            c

            for c in df.columns

            if c != target

            and pd.api.types.is_numeric_dtype(
                df[c]
            )

        ]

        for col in numeric_cols:

            try:

                corr = df[
                    [col, target]
                ].corr().iloc[0, 1]

                correlations.append({

                    "feature":
                    col,

                    "correlation":
                    round(
                        corr,
                        3
                    )

                })

            except:
                pass

        correlations = sorted(

            correlations,

            key=lambda x:
            abs(
                x["correlation"]
            ),

            reverse=True

        )

        driver_analysis = []

        for col in numeric_cols:

            try:

                corr = df[
                    [col, target]
                ].corr().iloc[0, 1]

                if pd.isna(corr):
                    continue

                median = df[col].median()

                high = df[
                    df[col] >= median
                ][target]

                low = df[
                    df[col] < median
                ][target]

                if len(high) < 10:
                    continue

                if len(low) < 10:
                    continue

                _, p_value = ttest_ind(
                    high,
                    low,
                    equal_var=False
                )

                driver_analysis.append({

                    "feature":
                    col,

                    "correlation":
                    round(
                        float(corr),
                        3
                    ),

                    "relationship":
                    (
                        "positive"
                        if corr > 0
                        else "negative"
                    ),

                    "strength":
                    round(
                        abs(
                            float(corr)
                        ),
                        3
                    ),

                    "p_value":
                    round(
                        float(
                            p_value
                        ),
                        5
                    )

                })

            except:
                pass

        driver_analysis = sorted(

            driver_analysis,

            key=lambda x: (

                abs(
                    x["correlation"]
                ),

                1 - x["p_value"]

            ),

            reverse=True

        )

        return {

            "driver_analysis":
            driver_analysis[:20],

            "correlations":
            correlations[:20],

            "target_summary": {

                "mean":
                round(
                    df[target].mean(),
                    2
                ),

                "median":
                round(
                    df[target].median(),
                    2
                ),

                "std":
                round(
                    df[target].std(),
                    2
                )

            }

        }