from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureTools:


    @staticmethod
    def remove_ids(df,schema):

        roles=schema[
        "column_roles"
        ]

        cols=[]

        for c,r in roles.items():

            if r=="identifier":

                cols.append(c)

        return df.drop(
            columns=cols,
            errors="ignore"
        )


    @staticmethod
    def process_dates(
        df,
        schema
    ):

        roles=schema[
        "column_roles"
        ]

        date_cols=[]

        for c,r in roles.items():

            if r=="datetime":

                date_cols.append(c)

        for col in date_cols:

            df[col]=pd.to_datetime(
                df[col],
                errors='coerce',
                format='mixed'
            )

            df[
            f"{col}_month"
            ]=(
                df[col]
                .dt.month
            )

            df[
            f"{col}_weekday"
            ]=(
                df[col]
                .dt.weekday
            )

            df[
            f"{col}_days_ago"
            ]=(

                pd.Timestamp.now()

                -

                df[col]

            ).dt.days

        return df.drop(
            columns=date_cols
        )
    @staticmethod
    def process_semantic_values(
        df,
        semantic_schema
    ):

        number_words={

            "one":1,
            "two":2,
            "three":3,
            "four":4,
            "five":5,
            "six":6,
            "seven":7,
            "eight":8,
            "nine":9,
            "ten":10
        }

        for col,info in semantic_schema.items():

            if col not in df.columns:
                continue

            series=(
                df[col]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            if info["semantic"]=="categorical":

                series=series.replace(
                    number_words
                )

                series=(
                    series
                    .str.extract(
                        r"(\d+)"
                    )[0]
                    .fillna(series)
                )

                true_values=[

                    "yes",
                    "y",
                    "true",
                    "available"

                ]

                false_values=[

                    "no",
                    "n",
                    "false",
                    "none"

                ]

                series=series.replace(
                    true_values,
                    "1"
                )

                series=series.replace(
                    false_values,
                    "0"
                )

                df[col]=series

        return df

    @staticmethod
    def encode(df, target):

        categorical = df.select_dtypes(
            include=['object']
        ).columns.tolist()

        # Remove target if present
        if target in categorical:
            categorical.remove(
                target
            )

        df = pd.get_dummies(
            df,
            columns=categorical,
            drop_first=True,
            dtype=int
        )

        return df


    @staticmethod
    def scale(df, target, schema=None):

        

        y = df[target]
        X = df.drop(columns=[target])

        scaler = StandardScaler()

        cols_to_scale = []

        numerical = X.select_dtypes(
            include=['number']
        )

        # Get schema roles if available
        roles = {}

        if schema:
            roles = schema.get(
                "column_roles",
                {}
            )

        for col in numerical.columns:

            unique_count = (
                X[col]
                .nunique()
            )

            # Skip binary / one-hot columns
            if unique_count <= 2:
                continue

            # Skip categorical features if schema knows them
            if (
                col in roles and
                roles[col] == "categorical"
            ):
                continue

            # Determine if feature behaves like continuous
            uniqueness_ratio = (
                unique_count / len(X)
            )

            # Scale continuous numeric features
            if uniqueness_ratio > 0.05:

                cols_to_scale.append(
                    col
                )

        print(
            "\nScaling columns:"
        )

        print(
            cols_to_scale
        )

        if cols_to_scale:

            X[
                cols_to_scale
            ] = (
                scaler.fit_transform(
                    X[
                        cols_to_scale
                    ]
                )
            )

        X[target] = y

        return X
    



    @staticmethod
    def encode_target(
        df,
        target
    ):

        if target not in df.columns:

            return df


        col=df[target]


        # Fill missing target values
        col=col.fillna(
            "Unknown"
        )


        # ---------------------
        # Numeric target
        # ---------------------

        if pd.api.types.is_numeric_dtype(
            col
        ):

            print(
                f"{target} detected as numeric target"
            )

            return df


        # ---------------------
        # Check cardinality
        # ---------------------

        unique_count=(

            col
            .nunique()

        )


        total_rows=(

            len(col)

        )


        unique_ratio=(

            unique_count/
            total_rows

        )


        # ---------------------
        # High cardinality
        # usually regression-like
        # ---------------------

        if unique_ratio>0.5:

            print(

                f"{target} appears continuous — skipping encoding"

            )

            return df


        # ---------------------
        # Classification target
        # ---------------------

        le=LabelEncoder()

        df[target]=(

            le.fit_transform(

                col.astype(
                    str
                )

            )

        )

        print(

            f"{target} label encoded"

        )

        print(

            dict(

                zip(
                    le.classes_,
                    le.transform(
                        le.classes_
                    )
                )

            )

        )


        return df
    
    @staticmethod
    def handle_missing(df):

        numeric=df.select_dtypes(
            include=['number']
        )

        categorical=df.select_dtypes(
            exclude=['number']
        )

        for col in numeric:

            df[col]=(
                df[col]
                .fillna(
                    df[col].median()
                )
            )

        for col in categorical:

            mode=df[col].mode()

            if len(mode)>0:

                df[col]=(
                    df[col]
                    .fillna(
                        mode[0]
                    )
                )

        return df
    

    @staticmethod
    def apply_smote(df, target):

        y = df[target]

        X = df.drop(
            columns=[target]
        )

        # SMOTE only for binary/multiclass targets
        if y.nunique() < 2:
            return df

        try:

            smote = SMOTE(
                random_state=42
            )

            X_resampled, y_resampled = (
                smote.fit_resample(
                    X,
                    y
                )
            )

            new_df = X_resampled.copy()

            new_df[target] = y_resampled

            return new_df

        except Exception as e:

            print(
                f"SMOTE skipped: {e}"
            )

            return df
        
    @staticmethod
    def apply_pca(df,target):

        y=df[target]

        X=df.drop(
            columns=[target]
        )

        # Only apply if many features exist
        if X.shape[1] < 8:

            return df

        try:

            pca=PCA(
                n_components=0.95
            )

            X_pca=(
                pca.fit_transform(
                    X
                )
            )

            cols=[

                f"PC{i+1}"

                for i in range(
                    X_pca.shape[1]
                )

            ]

            pca_df=pd.DataFrame(
                X_pca,
                columns=cols
            )

            pca_df[target]=(
                y.values
            )

            return pca_df

        except Exception as e:

            print(
                f"PCA skipped: {e}"
            )

            return df
        
    @staticmethod
    def normalize_categories(df):

        categorical = df.select_dtypes(
            include=['object']
        )

        for col in categorical:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        return df
    



    @staticmethod
    def process_cyclical_features(df):


        cyclical_columns={

            "month":12,
            "weekday":7,
            "hour":24,
            "quarter":4

        }


        for col in df.columns:

            col_lower=col.lower()


            for keyword,max_value in (

                cyclical_columns.items()

            ):

                if keyword in col_lower:


                    # force numeric

                    numeric=(
                        pd.to_numeric(
                            df[col],
                            errors="coerce",
                            
                        )
                    )


                    # skip if conversion fails

                    if numeric.notna().mean()<0.7:

                        print(
                            f"Skipping cyclical {col}"
                        )

                        continue


                    df[
                        f"{col}_sin"
                    ]=(

                        np.sin(

                            2*np.pi*
                            numeric/
                            max_value

                        )

                    )


                    df[
                        f"{col}_cos"
                    ]=(

                        np.cos(

                            2*np.pi*
                            numeric/
                            max_value

                        )

                    )


                    df.drop(
                        col,
                        axis=1,
                        inplace=True
                    )

                    break


        return df
    
    @staticmethod
    def process_semantic_columns(
        df,
        semantic_schema
    ):

        for col,info in (

            semantic_schema.items()

        ):

            semantic=info[
                "semantic"
            ]


            # currency cleanup

            if semantic=="currency":

                df[col]=(

                    df[col]
                    .astype(str)

                    .str.replace(
                        r"[₹,$,€]",
                        "",
                        regex=True
                    )

                )


            # percentage cleanup

            elif semantic=="percentage":

                df[col]=(

                    df[col]
                    .astype(str)

                    .str.replace(
                        "%",
                        ""
                    )

                )


        return df
    @staticmethod
    def convert_semantic_numeric(
        df,
        semantic_schema
    ):

        numeric_semantics=[

            "currency",
            "distance",
            "area",
            "percentage"

        ]

        for col,info in semantic_schema.items():

            if col not in df.columns:
                continue

            if info["semantic"] in numeric_semantics:

                df[col]=(

                    df[col]
                    .astype(str)

                    .str.replace(
                        r"[^\d.]",
                        "",
                        regex=True
                    )

                )

                df[col]=pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        return df