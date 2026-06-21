from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class SegmentationAnalyzer:

    @staticmethod
    def analyze(
        df,
        target
    ):

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if target in numeric_cols:

            numeric_cols.remove(
                target
            )

        if len(numeric_cols) < 2:

            return {

                "segments": [],

                "best_segment": None

            }

        # ====================================
        # Prepare Data
        # ====================================

        X = df[
            numeric_cols
        ].fillna(
            df[numeric_cols].median()
        )

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        # ====================================
        # Dynamic Cluster Count
        # ====================================

        n_clusters = min(

            4,

            max(
                2,
                len(df) // 100
            )

        )

        kmeans = KMeans(

            n_clusters=n_clusters,

            random_state=42,

            n_init=20

        )

        clusters = kmeans.fit_predict(
            X_scaled
        )

        temp = df.copy()

        temp[
            "cluster"
        ] = clusters

        segments = []

        # ====================================
        # Build Segment Profiles
        # ====================================

        overall_target = (
            temp[target]
            .mean()
        )

        for cluster in sorted(
            temp["cluster"].unique()
        ):

            group = temp[
                temp["cluster"]
                == cluster
            ]

            avg_target = round(

                group[target]
                .mean(),

                2

            )

            uplift = round(

                (
                    (
                        avg_target
                        -
                        overall_target
                    )

                    /

                    max(
                        abs(
                            overall_target
                        ),
                        1
                    )

                ) * 100,

                2

            )

            characteristics = {}

            for feature in numeric_cols:

                characteristics[
                    feature
                ] = round(

                    group[
                        feature
                    ].mean(),

                    2

                )

            # Top distinguishing features

            feature_deltas = []

            for feature in numeric_cols:

                overall_mean = (
                    temp[
                        feature
                    ].mean()
                )

                segment_mean = (
                    group[
                        feature
                    ].mean()
                )

                overall_std = temp[feature].std()

                if overall_std == 0:
                    continue

                delta = abs(
                    segment_mean -
                    overall_mean
                ) / overall_std

                feature_deltas.append(

                    (
                        feature,
                        delta
                    )

                )

            feature_deltas = sorted(

                feature_deltas,

                key=lambda x: x[1],

                reverse=True

            )[:5]

            top_characteristics = [

                feature

                for feature, _

                in feature_deltas

            ]

            segment_name = (
                " / ".join(
                    top_characteristics[:3]
                )
            )

            segments = sorted(
                segments,
                key=lambda x: x["avg_target"],
                reverse=True
            )

            best_segment = (
                segments[0]
                if segments
                else None
            )

            segments.append({

                "segment":
                f"Cluster {cluster}",

                "records":
                len(group),

                "percentage":
                round(

                    (
                        len(group)
                        /
                        len(temp)
                    ) * 100,

                    2

                ),

                "avg_target":
                avg_target,

                "uplift_pct":
                uplift,

                "segment_summary": {
                feature:
                round(group[feature].mean(),2)
                for feature in top_characteristics
            },

                "segment_name":
                segment_name,
                "top_characteristics": 
                top_characteristics


            })




        return {

            "segments":
            segments,

            "best_segment":
            best_segment

        }