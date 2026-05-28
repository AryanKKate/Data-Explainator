import pandas as pd

class DatasetProfiler:

    @staticmethod
    def profile(df, target):

        X=df.drop(columns=[target])

        profile={}

        profile["rows"]=len(df)

        profile["cols"]=len(df.columns)

        profile["feature_count"]=len(X.columns)

        profile["missing_ratio"]=(
            df.isnull().sum().sum()
            /
            (df.shape[0]*df.shape[1])
        )

        numeric_cols=(
            X.select_dtypes(
                include=["number"]
            ).columns
        )

        categorical_cols=(
            X.select_dtypes(
                exclude=["number"]
            ).columns
        )

        profile["numeric_features"]=len(numeric_cols)

        profile["categorical_features"]=len(categorical_cols)

        profile["high_dimensional"]=(
            len(X.columns)>100
        )

        profile["small_dataset"]=(
            len(df)<1000
        )

        profile["large_dataset"]=(
            len(df)>100000
        )

        profile["sparse_dataset"]=(
            profile["missing_ratio"]>0.3
        )

        target_unique=df[target].nunique()

        if target_unique<=10:

            imbalance_ratio=(
                df[target]
                .value_counts(normalize=True)
                .min()
            )

            profile["imbalanced"]=(
                imbalance_ratio<0.2
            )

        else:

            profile["imbalanced"]=False

        return profile