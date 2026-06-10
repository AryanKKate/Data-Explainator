from utils.llm import llm

from pydantic import BaseModel
from typing import List


class RecommendationOutput(BaseModel):

    recommendations: List[str]

    opportunities: List[str]

    risks: List[str]

    quick_wins: List[str]


class RecommendationAgent:

    @staticmethod
    def generate(
        task,
        target,
        insights,
        semantic_schema,
        training_results,
        query,
        explainability_report,
        validation_report,
        business_intelligence,
        visualizations
    ):

        best_model = training_results.get(
            "best_model",
            ""
        )

        metrics = (

            training_results
            .get("results", {})
            .get(best_model, {})

        )

        prompt = f"""

You are a Principal Data Scientist and Strategy Consultant.

Your job is NOT to explain the model.

Your job is to identify:

1. Business Opportunities
2. Operational Risks
3. Strategic Recommendations
4. Quick Wins

using the analytical evidence provided.

------------------------------------------------
USER QUESTION
------------------------------------------------

{query}

------------------------------------------------
TASK
------------------------------------------------

{task}

------------------------------------------------
TARGET
------------------------------------------------

{target}

------------------------------------------------
MODEL
------------------------------------------------

{best_model}

------------------------------------------------
MODEL PERFORMANCE
------------------------------------------------

{metrics}

------------------------------------------------
INSIGHTS
------------------------------------------------

{insights}

------------------------------------------------
EXPLAINABILITY
------------------------------------------------

{explainability_report}

------------------------------------------------
DATA QUALITY
------------------------------------------------

{validation_report}

------------------------------------------------
SEMANTIC SCHEMA
------------------------------------------------

{semantic_schema}

------------------------------------------------
INSTRUCTIONS
------------------------------------------------

Generate:

recommendations:
- 5 strategic recommendations

opportunities:
- 3 business opportunities

risks:
- 3 important risks

quick_wins:
- 3 high-impact actions

Requirements:

- Must be evidence driven.
- Use SHAP drivers.
- Use model quality.
- Use dataset characteristics.
- Use business context.
- Never invent numbers.
- Never mention machine learning jargon.
- Speak like a McKinsey/Bain consultant.
- Focus on business value.

"""

        structured_llm = (

            llm.with_structured_output(
                RecommendationOutput
            )

        )

        result = structured_llm.invoke(
            prompt
        )

        return result.model_dump()