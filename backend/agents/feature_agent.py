from tools.preprocessing_tools import FeatureTools


class FeatureAgent:

    @staticmethod
    def process(
        df,
        steps,
        schema,
        target
    ):

        print("\n===== FeatureAgent Started =====")

        # Remove identifiers
        df = FeatureTools.remove_ids(
            df,
            schema
        )

        print(
            "After remove_ids:",
            df.shape
        )

        # Process datetime columns
        df = FeatureTools.process_dates(
            df,
            schema
        )

        print(
            "After process_dates:",
            df.shape
        )

        # Convert cyclical features
        df = FeatureTools.process_cyclical_features(
            df
        )

        print(
            "After cyclical processing:",
            df.shape
        )

        # Missing values
        df = FeatureTools.handle_missing(
            df
        )

        print(
            "After missing handling:",
            df.shape
        )

        # Normalize categories
        df = FeatureTools.normalize_categories(
            df
        )

        print(
            "After category normalization:",
            df.shape
        )

        # Encode target separately
        if target in df.columns:

            df = FeatureTools.encode_target(
                df,
                target
            )

        print(
            "After target encoding:",
            df.shape
        )

        # Planned transformations
        for step in steps:

            print(
                f"\nExecuting: {step}"
            )

            if step == "encoding":

                df = FeatureTools.encode(
                    df,
                    target
                )

            elif step == "scaling":

                df = FeatureTools.scale(
                    df,
                    target,
                    schema
                )

            elif step == "smote":

                df = FeatureTools.apply_smote(
                    df,
                    target
                )

            elif step == "pca":

                df = FeatureTools.apply_pca(
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

        print("\n===== FeatureAgent Finished =====")

        return df