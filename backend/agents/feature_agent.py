from tools.preprocessing_tools import FeatureTools


class FeatureAgent:

    @staticmethod
    def process(
        df,
        steps,
        schema,
        semantic_schema,
        target
    ):

        print("\n===== FeatureAgent Started =====")


        df=df.copy()



        df=FeatureTools.remove_ids(
            df,
            schema
        )

        print(
            "After remove_ids:",
            df.shape
        )



        df=FeatureTools.process_semantic_values(
    df,
    semantic_schema
)

        print(
            "After semantic processing:",
            df.shape
        )

        df=FeatureTools.convert_semantic_numeric(
            df,
            semantic_schema
        )

        print(
        "After semantic numeric conversion:",
        df.shape
        )



        df=FeatureTools.process_dates(
            df,
            schema
        )

        print(
            "After process_dates:",
            df.shape
        )



        df=FeatureTools.process_cyclical_features(
            df
        )

        print(
            "After cyclical processing:",
            df.shape
        )

        df=FeatureTools.handle_missing(
            df
        )

        print(
            "After missing handling:",
            df.shape
        )



        df=FeatureTools.normalize_categories(
            df
        )

        print(
            "After category normalization:",
            df.shape
        )



        if (

            target
            and
            target in df.columns

        ):

            df=FeatureTools.encode_target(
                df,
                target
            )

        print(
            "After target encoding:",
            df.shape
        )


        for step in steps:

            print(
                f"\nExecuting: {step}"
            )


            if step=="encoding":

                df=FeatureTools.encode(
                    df,
                    target
                )


            elif step=="scaling":

                df=FeatureTools.scale(
                    df,
                    target,
                    schema
                )


            elif step=="smote":

                if (

                    target
                    and
                    target in df.columns

                ):

                    df=FeatureTools.apply_smote(
                        df,
                        target
                    )


            elif step=="pca":

                if (

                    target
                    and
                    target in df.columns

                ):

                    df=FeatureTools.apply_pca(
                        df,
                        target
                    )


            else:

                print(
                    f"Unknown step skipped: {step}"
                )


            print(
                "Current shape:",
                df.shape
            )


        print(
            "\n===== FeatureAgent Finished ====="
        )

        return df