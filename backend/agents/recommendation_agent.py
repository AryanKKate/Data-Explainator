from utils.llm import llm
from pydantic import BaseModel
from typing import List


class RecommendationOutput(BaseModel):

    recommendations: List[str]


class RecommendationAgent:

    @staticmethod
    def generate(
        task,
        target,
        insights,
        semantic_schema,
        training_results,
        query
    ):

        best_model = training_results.get(
            "best_model",
            ""
        )

        model_metrics = (
            training_results
            .get("results", {})
            .get(best_model, {})
        )

        prompt = f"""
You are a Senior Data Analyst.

Task:
{task}

Target:
{target}

Insights:
{insights}

Semantic Schema:
{semantic_schema}

Best Model:
{best_model}

Metrics:
{model_metrics}

User has asked the following query. Based on the query, insights, and model performance, generate 5 actionable business recommendations.:
{query}


Rules:

- Recommendations must be specific.
- Recommendations must be data-driven.
- Use insights and top drivers.
- Avoid generic advice.
- Keep each recommendation under 20 words.
- Focus on business impact.

Examples:

House Pricing:
- Target high-income neighborhoods.
- Focus marketing in high population regions.

Inventory:
- Replenish products with recurring low stock.
- Review suppliers linked to stockouts.

Customer Churn:
- Retain customers with declining engagement.
- Investigate high-risk customer segments.
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