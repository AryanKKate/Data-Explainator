from pydantic import BaseModel
from typing import List

from utils.llm import llm


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
        training_results,
        explainability_report,
        business_intelligence,
        query

    ):

        # =====================================
        # Model Metrics
        # =====================================

        best_model = training_results[
            "best_model"
        ]

        metrics = training_results[
            "results"
        ][best_model]

        # =====================================
        # SHAP Drivers
        # =====================================

        importance = (
            explainability_report.get(
                "global_importance_percentages",
                {}
            )
        )

        # =====================================
        # Opportunities
        # =====================================

        opportunity_ranking = (

            business_intelligence.get(
                "opportunity_ranking",
                []
            )

        )

        evidence = []

        for item in opportunity_ranking[:10]:

            feature = item.get(
                "feature"
            )

            evidence.append({

                "feature":
                feature,

                "importance":
                round(
                    importance.get(
                        feature,
                        0
                    ),
                    2
                ),

                "opportunity_score":
                round(
                    item.get(
                        "opportunity_score",
                        0
                    ),
                    2
                )

            })

        # =====================================
        # Segments
        # =====================================

        segment_data = (

            business_intelligence
            .get(
                "segments",
                {}
            )
            .get(
                "segments",
                []
            )

        )

        # =====================================
        # Trust
        # =====================================

        trust_score = (

            business_intelligence
            .get(
                "trust_score",
                {}
            )
            .get(
                "trust_score",
                0
            )

        )

        # =====================================
        # Data Quality
        # =====================================

        data_quality = (

            business_intelligence
            .get(
                "data_quality",
                {}
            )
            .get(
                "overall_score",
                0
            )

        )

        # =====================================
        # Prompt
        # =====================================

        prompt = f"""

You are a Senior Analytics Consultant.

Your job is to generate executive business recommendations based on user query{query} and the results of a comprehensive data analysis

IMPORTANT RULES:

- Use ONLY evidence provided.
- Never invent domain facts.
- Never assume industry context.
- Never mention houses, customers, products,
  patients, employees, etc unless present in evidence.
- Recommendations must be data-driven.
- Recommendations must be supported by metrics.

----------------------------------------

TASK

{task}

TARGET

{target}

----------------------------------------

MODEL

Best Model:
{best_model}

Metrics:
{metrics}

Trust Score:
{trust_score}

Data Quality Score:
{data_quality}

----------------------------------------

TOP OPPORTUNITIES

{evidence}

----------------------------------------

SEGMENTS

{segment_data}

----------------------------------------

Generate:

1. 5 Recommendations
2. 3 Opportunities
3. 3 Risks
4. 3 Quick Wins

Rules:

Recommendations:
- Actionable
- Executive level
- Under 25 words

Opportunities:
- Based only on highest opportunity scores

Risks:
- Based on model reliability
- Based on data quality
- Based on concentration of drivers

Quick Wins:
- Easy actions based on top opportunities

Return concise bullet style output.

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