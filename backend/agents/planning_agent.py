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

Return a single JSON object with a key named "steps".
The value of "steps" must be a JSON array of strings.
Do not include any text outside the JSON object.

Mapping rules:
- classification -> ["encoding", "scaling", "smote (if imbalance exists)"]
- regression -> ["encoding", "scaling"]
- forecasting -> ["date_features"]
- exploratory_analysis -> ["encoding", "scaling"]
- clustering -> ["encoding", "scaling"]

If the task is unknown, return ["encoding", "scaling"].

Example output:
{{"steps": ["encoding", "scaling"]}}

"""

        result=structured_llm.invoke(
            prompt
        )

        return result.model_dump()