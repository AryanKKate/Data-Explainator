from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import pandas as pd


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

        from sklearn.preprocessing import StandardScaler

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
    def encode_target(df,target):

        le=LabelEncoder()

        df[target]=(
            le.fit_transform(
                df[target]
                .astype(str)
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

        import numpy as np

        weekday_map = {

            "mon":0,
            "monday":0,

            "tue":1,
            "tuesday":1,

            "wed":2,
            "wednesday":2,

            "thu":3,
            "thursday":3,

            "fri":4,
            "friday":4,

            "sat":5,
            "saturday":5,

            "sun":6,
            "sunday":6
        }

        month_map = {

            "jan":1,
            "january":1,

            "feb":2,
            "february":2,

            "mar":3,
            "march":3,

            "apr":4,
            "april":4,

            "may":5,

            "jun":6,
            "june":6,

            "jul":7,
            "july":7,

            "aug":8,
            "august":8,

            "sep":9,
            "september":9,

            "oct":10,
            "october":10,

            "nov":11,
            "november":11,

            "dec":12,
            "december":12
        }

        for col in df.columns:

            col_lower=col.lower()

            # Weekday handling

            if "weekday" in col_lower:

                if df[col].dtype=="object":

                    df[col]=(
                        df[col]
                        .astype(str)
                        .str.lower()
                        .map(
                            weekday_map
                        )
                    )

                radians=(
                    2*np.pi*
                    df[col]/7
                )

                df[
                    f"{col}_sin"
                ]=np.sin(
                    radians
                )

                df[
                    f"{col}_cos"
                ]=np.cos(
                    radians
                )

                df.drop(
                    columns=[col],
                    inplace=True
                )

            # Month handling

            elif "month" in col_lower:

                if df[col].dtype=="object":

                    df[col]=(
                        df[col]
                        .astype(str)
                        .str.lower()
                        .map(
                            month_map
                        )
                    )

                radians=(
                    2*np.pi*
                    df[col]/12
                )

                df[
                    f"{col}_sin"
                ]=np.sin(
                    radians
                )

                df[
                    f"{col}_cos"
                ]=np.cos(
                    radians
                )

                df.drop(
                    columns=[col],
                    inplace=True
                )

        return df