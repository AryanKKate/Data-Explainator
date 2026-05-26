class ValidationAgent:

    @staticmethod
    def validate(df):

        report={}

        report[
        "missing"
        ]=(
            df.isnull()
            .sum()
            .sum()
        )

        report[
        "duplicates"
        ]=(
            df.duplicated()
            .sum()
        )

        report[
        "columns"
        ]=list(
            df.columns
        )

        return report