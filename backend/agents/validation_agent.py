class ValidationAgent:

    @staticmethod
    def validate(df):

        report = {}

        report["missing"] = int(
            df.isnull().sum().sum()
        )

        report["duplicates"] = int(
            df.duplicated().sum()
        )

        report["columns"] = list(
            df.columns
        )

        return report