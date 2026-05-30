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

        df = df.copy()

        # =====================================
        # Remove identifiers
        # =====================================

        df = FeatureTools.remove_ids(
            df,
            schema
        )

        print(
            "After remove_ids:",
            df.shape
        )

        # =====================================
        # Semantic cleaning
        # =====================================

        df = FeatureTools.process_semantic_values(
            df,
            semantic_schema
        )

        print(
            "After semantic processing:",
            df.shape
        )

        # =====================================
        # Convert semantic numerics
        # =====================================

        df = FeatureTools.convert_semantic_numeric(
            df,
            semantic_schema
        )

        print(
            "After semantic numeric conversion:",
            df.shape
        )

        # =====================================
        # Process dates
        # =====================================

        df = FeatureTools.process_dates(
            df,
            schema
        )

        print(
            "After process_dates:",
            df.shape
        )

        # =====================================
        # Cyclical features
        # =====================================

        df = FeatureTools.process_cyclical_features(
            df
        )

        print(
            "After cyclical processing:",
            df.shape
        )

        # =====================================
        # Missing values
        # =====================================

        df = FeatureTools.handle_missing(
            df
        )

        print(
            "After missing handling:",
            df.shape
        )

        # =====================================
        # Normalize categories
        # =====================================

        df = FeatureTools.normalize_categories(
            df
        )

        print(
            "After category normalization:",
            df.shape
        )

        # =====================================
        # Encode target
        # =====================================

        if (

            target
            and
            target in df.columns

        ):

            df = FeatureTools.encode_target(
                df,
                target
            )

        print(
            "After target encoding:",
            df.shape
        )

        # =====================================
        # ONLY SAFE preprocessing here
        # =====================================

        for step in steps:

            print(
                f"\nExecuting: {step}"
            )

            # =================================
            # Encoding
            # =================================

            if step == "encoding":

                df = FeatureTools.encode(
                    df,
                    target
                )

            # =================================
            # SKIP scaling here
            # =================================

            elif step == "scaling":

                print(
                    "Scaling deferred to TrainingAgent"
                )

            # =================================
            # SKIP SMOTE here
            # =================================

            elif step == "smote":

                print(
                    "SMOTE deferred to TrainingAgent"
                )

            # =================================
            # SKIP PCA here
            # =================================

            elif step == "pca":

                print(
                    "PCA deferred to TrainingAgent"
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

        import os

        os.makedirs("data", exist_ok=True)

        df.to_csv(
            "data/preprocessed_data.csv",
            index=False
        )

        print(
            "\nSaved preprocessed data to data/preprocessed_data.csv"
        )

        return df