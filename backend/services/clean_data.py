import pandas as pd

class DataCleaner:

    @staticmethod
    def clean(df):

        df=df.drop_duplicates()

        numerical=df.select_dtypes(
            include=['number']
        )

        categorical=df.select_dtypes(
            exclude=['number']
        )

        for col in numerical:
            df[col]=df[col].fillna(
                df[col].median()
            )

        for col in categorical:
            df[col]=df[col].fillna(
                df[col].mode()[0]
            )

        return df