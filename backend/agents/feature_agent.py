from tools.preprocessing_tools import FeatureTools


class FeatureAgent:

    @staticmethod
    def process(df,steps):


        for step in steps:

            if step=="encoding":

                df=FeatureTools.encode(
                    df
                )


            elif step=="scaling":

                df=FeatureTools.scale(
                    df
                )

        return df