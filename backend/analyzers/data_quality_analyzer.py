import pandas as pd
import numpy as np


class DataQualityAnalyzer:

    @staticmethod
    def analyze(df):

        rows = len(df)

        missing_columns = {}
        quality_warnings = []

        # -----------------------
        # Missing values
        # -----------------------

        for col in df.columns:

            pct = round(
                df[col].isna().mean() * 100,
                2
            )

            if pct > 0:

                missing_columns[col] = pct

            if pct > 20:

                quality_warnings.append(
                    f"{col} has high missingness ({pct}%)"
                )

        missing_score = max(
            0,
            100 - (
                sum(missing_columns.values())
                /
                max(len(df.columns), 1)
            )
        )

        # -----------------------
        # Duplicates
        # -----------------------

        duplicate_pct = round(

            df.duplicated().mean()
            * 100,

            2

        )

        duplicate_score = max(
            0,
            100 - duplicate_pct
        )

        # -----------------------
        # Constant Columns
        # -----------------------

        constant_columns = [

            c

            for c in df.columns

            if df[c].nunique(dropna=False) <= 1

        ]

        # -----------------------
        # High Cardinality
        # -----------------------

        high_cardinality = []

        for col in df.columns:

            ratio = (

                df[col]
                .nunique()

                /

                max(rows, 1)

            )

            if ratio > 0.5:

                high_cardinality.append(col)

        # -----------------------
        # Outliers
        # -----------------------

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        outlier_count = 0

        total_numeric = 0

        for col in numeric_cols:

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = (

                (df[col] < lower)

                |

                (df[col] > upper)

            ).sum()

            outlier_count += outliers

            total_numeric += len(df)

        outlier_pct = round(

            (
                outlier_count
                /
                max(total_numeric, 1)
            ) * 100,

            2

        )

        outlier_score = max(
            0,
            100 - outlier_pct
        )

        # -----------------------
        # Overall
        # -----------------------

        overall_score = round(

            (
                missing_score +
                duplicate_score +
                outlier_score
            ) / 3,

            1

        )

        return {

            "overall_score":
            overall_score,

            "missing_score":
            round(missing_score, 1),

            "duplicate_score":
            round(duplicate_score, 1),

            "outlier_score":
            round(outlier_score, 1),

            "missing_columns":
            missing_columns,

            "constant_columns":
            constant_columns,

            "high_cardinality":
            high_cardinality,

            "quality_warnings":
            quality_warnings

        }