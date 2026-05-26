from utils.llm import llm
from pydantic import BaseModel
from typing import Dict


class SchemaOutput(BaseModel):

    column_roles: Dict[str,str]


class SchemaAgent:

    @staticmethod
    def analyze(columns):

        structured_llm = (
            llm.with_structured_output(
                SchemaOutput
            )
        )

        prompt=f"""

Determine the role of each column.

Columns:

{columns}

Possible roles:

identifier
numerical
categorical
datetime
target

Rules(examples):

- customer_id,id,user_id → identifier
- date,time,last_login → datetime
- churn,target,label → target

Return role for every column.

"""

        result=structured_llm.invoke(
            prompt
        )

        return result.model_dump()