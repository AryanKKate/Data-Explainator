from tools.preprocessing_tools import FeatureTools 



class FeatureAgent:


    @staticmethod
    def process(
        df,
        steps,
        schema,
        target
    ):


        df=FeatureTools.remove_ids(
            df,
            schema
        )


        df=FeatureTools.process_dates(
            df,
            schema
        )

        df=FeatureTools.handle_missing(
            df
        )       

        for step in steps:


            if step=="encoding":

                df=FeatureTools.encode(
                    df,
                    target
                )


            elif step=="scaling":

                df=FeatureTools.scale(
                    df,
                    target
                )


        return df