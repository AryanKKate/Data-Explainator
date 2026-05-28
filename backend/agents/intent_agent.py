from utils.llm import llm

from pydantic import BaseModel

from difflib import get_close_matches


class IntentOutput(BaseModel):

    task_type: str

    target: str | None = None


class IntentAgent:

    @staticmethod
    def match_target_column(

        extracted_target,

        columns

    ):

        if extracted_target is None:

            return None

        extracted_target = (

            extracted_target
            .lower()
            .strip()

        )

        normalized_columns = {

            col.lower(): col

            for col in columns

        }

        # Exact match

        if extracted_target in normalized_columns:

            return normalized_columns[
                extracted_target
            ]

        # Token similarity match

        matches = get_close_matches(

            extracted_target,

            normalized_columns.keys(),

            n=1,

            cutoff=0.4

        )

        if matches:

            return normalized_columns[
                matches[0]
            ]

        # Partial semantic fallback

        for col in columns:

            col_lower = col.lower()

            if (

                extracted_target in col_lower

                or

                col_lower in extracted_target

            ):

                return col

        return None


    @staticmethod
    def detect(

        query,

        columns

    ):

        structured_llm = (

            llm.with_structured_output(

                IntentOutput

            )

        )

        prompt = f"""
        Determine:

        1. task type:
            - classification
            - regression
            - clustering
            - forecasting
            - anomaly_detection
            - exploratory_analysis

        2. target column name from the dataset

        IMPORTANT:
        - Return the most likely REAL column name
        - Infer semantic meaning
        - Example:
            "predict inventory status"
            -> "Status"

        Available columns:

        {list(columns)}

        User Query:

        {query}
        """

        result = structured_llm.invoke(
            prompt
        )

        output = result.model_dump()

        matched_target = (

            IntentAgent.match_target_column(

                extracted_target=output["target"],

                columns=columns

            )

        )

        output["target"] = matched_target

        return output