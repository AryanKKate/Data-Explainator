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

        df = FeatureTools.encode_target(
            df,
            target
        )   

        df = FeatureTools.normalize_categories(
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


            elif step=="smote":

                df=FeatureTools.apply_smote(
                    df,
                    target
                )


            elif step=="pca":

                df=FeatureTools.apply_pca(
                    df,
                    target
                )


        return df