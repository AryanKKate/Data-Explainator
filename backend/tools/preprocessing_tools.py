from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import pandas as pd


class FeatureTools:


    @staticmethod
    def encode(df):

        categorical=df.select_dtypes(
            include=['object']
        )

        for col in categorical:

            le=LabelEncoder()

            df[col]=le.fit_transform(
                df[col].astype(str)
            )

        return df


    @staticmethod
    def scale(df):

        numerical=df.select_dtypes(
            include=['number']
        )

        scaler=StandardScaler()

        df[numerical.columns]=(
            scaler.fit_transform(
                numerical
            )
        )

        return df