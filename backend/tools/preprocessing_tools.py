from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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
    def scale(
        df,
        target
    ):

        y=df[target]

        X=df.drop(
            columns=[target]
        )

        numerical=(
            X.select_dtypes(
                include=['number']
            )
        )

        scaler=(
            StandardScaler()
        )

        X[
        numerical.columns
        ]=(

        scaler.fit_transform(
        numerical
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