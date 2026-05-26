class DataProfiler:

    @staticmethod
    def profile(df):

        return {

            "shape":df.shape,

            "columns":
            list(df.columns),

            "missing":
            df.isnull().sum().to_dict(),

            "dtypes":
            df.dtypes.astype(
            str
            ).to_dict()

        }