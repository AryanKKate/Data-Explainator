from utils.llm import llm
from pydantic import BaseModel


class IntentOutput(BaseModel):

    task_type: str
    target: str


class IntentAgent:

    @staticmethod
    def detect(query):

        structured_llm = llm.with_structured_output(
            IntentOutput
        )

        prompt = f"""
        Determine:

        - task type:
            classification
            regression
            clustering
            forecasting
            anomaly_detection
            exploratory_analysis

        - target column

        Query:

        {query}
        """

        result = structured_llm.invoke(
            prompt
        )

        return result.model_dump()