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

Choose from:

- missing_values
- encoding
- scaling
- smote
- date_features
- lag_features
- pca

"""

        result=structured_llm.invoke(
            prompt
        )

        return result.model_dump()