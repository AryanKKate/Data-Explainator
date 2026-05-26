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
    def encode(
        df,
        target
    ):

        categorical=(
            df.select_dtypes(
                include=['object']
            )
        )

        for col in categorical:

            if col!=target:

                le=LabelEncoder()

                df[col]=(
                    le.fit_transform(
                    df[col]
                    .astype(str)
                    )
                )

        return df


    @staticmethod
    def scale(df, target):

        y = df[target]

        X = df.drop(
            columns=[target]
        )

        scaler = StandardScaler()

        numerical = X.select_dtypes(
            include=['number']
        )

        cols_to_scale=[]

        for col in numerical.columns:

            # Ignore binary columns
            unique_count=(
                X[col]
                .nunique()
            )

            if unique_count > 2:

                cols_to_scale.append(
                    col
                )

        if cols_to_scale:

            X[
                cols_to_scale
            ]=(
                scaler.fit_transform(
                    X[
                        cols_to_scale
                    ]
                )
            )

        X[target]=y

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