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

        statistical_findings = (

            business_intelligence
            .get(
                "statistical_findings",
                {}
            )
            .get(
                "driver_analysis",
                []
            )

        )

        # =====================================
        # Opportunities
        # =====================================

        translated_insights = (

            business_intelligence.get(
                "translated_insights",
                []
            )

        )


        evidence = []

        for item in translated_insights[:10]:

            evidence.append({

                "feature":
                item.get(
                    "feature"
                ),

                "finding":
                item.get(
                    "finding"
                ),

                "importance":
                item.get(
                    "importance"
                ),

                "priority_score":
                item.get(
                    "priority_score"
                ),

                "actionable":
                item.get(
                    "actionable"
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
        top_drivers = explainability_report.get(
            "top_drivers",
            []
        )
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

TRANSLATED BUSINESS INSIGHTS

{evidence}

IMPORTANT:

- Never recommend increasing feature importance.
- Never recommend modifying model features.
- Never recommend improving a variable unless it is marked actionable=True.
- Use actionable=False items only as supporting evidence.
- Recommendations must focus on business decisions, not machine learning features.
- If no actionable opportunities exist, provide strategic monitoring recommendations instead.

----------------------------------------

SEGMENTS

{segment_data}

TOP MODEL DRIVERS

{top_drivers}

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

- Mention a risk ONLY if evidence supports it.
- Do not assume overfitting.
- Do not invent data quality problems.
- Do not invent reliability problems.

Examples:

Trust Score < 50
→ reliability risk

Data Quality < 70
→ data quality risk

Top driver concentration > 60%
→ dependency risk

Otherwise return:
"No major analytical risks identified"

Quick Wins:

- Must reference specific features.
- Must be supported by evidence.
- Must never mention model engineering.
- Must never suggest collecting more data.
- Must never suggest feature engineering.

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