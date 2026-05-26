from utils.llm import llm
from pydantic import BaseModel
from typing import List


class PlanOutput(BaseModel):

    steps: List[str]


class PlanningAgent:

    @staticmethod
    def create_plan(task):

        structured_llm = llm.with_structured_output(
            PlanOutput
        )

        prompt=f"""

Task:

{task}

Rules:

- Classification:
    encoding
    scaling
    smote (if imbalance exists)

- Regression:
    encoding
    scaling

- Forecasting:
    date_features

- PCA:
    only if feature count > 8

Return steps only

"""

        result=structured_llm.invoke(
            prompt
        )

        return result.model_dump()