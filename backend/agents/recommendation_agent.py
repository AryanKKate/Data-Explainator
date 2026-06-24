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
        query,
        insights

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

        business_observations = (
            business_intelligence.get(
                "business_observations",
                []
            )
        )
        business_observations = insights.get(
            "business_observations",
            []
        )





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

        ranked_insights = (
            business_intelligence.get(
                "ranked_insights",
                []
            )
        )
        opportunities = (
        business_intelligence.get(
            "opportunities",
            []
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
OBSERVATIONS
{business_observations}

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
BUSINESS OBSERVATIONS

{business_observations}

IMPORTANT:

IMPORTANT

Use business observations as the
primary evidence source.

Recommendations should focus on:

- monitoring
- prioritization
- segmentation
- investigation
- resource allocation
- risk mitigation

Do not recommend changing variables
unless evidence explicitly supports it.
- Use actionable=False items only as supporting evidence.
- Recommendations must focus on business decisions, not machine learning features.
- If no actionable opportunities exist, provide strategic monitoring recommendations instead.

----------------------------------------

SEGMENTS

{segment_data}

TOP RANKED INSIGHTS

{ranked_insights[:10]}

RANKED OPPORTUNITIES

{opportunities}

----------------------------------------

CRITICAL:

Features are explanatory variables.

A feature being important DOES NOT imply
it should be increased, decreased,
optimized, invested in, or targeted.

Never generate recommendations of the form:

- Increase X
- Reduce X
- Invest in X
- Improve X
- Optimize X

unless the evidence explicitly states
that such an action is possible.

Generate:

1. Recommendations
2. Risks
3. Quick Wins

Do NOT generate new opportunities.

Opportunities have already been identified by the analytical pipeline.
Only summarize the highest-ranked opportunities provided.

Rules:

Recommendations:
- Actionable
- Executive level
- Under 25 words
Recommendations must originate from:

- business observations
- ranked opportunities
- segment findings

Do not generate recommendations directly from feature names.
Opportunities:
- Based only on highest opportunity scores

Otherwise return:

["No major analytical risks identified"]

IMPORTANT:
recommendations, opportunities, risks, and quick_wins
must ALWAYS be arrays of strings.
Never return a single string.

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



Quick Wins:

- Must reference evidence.
- Do not assume a feature can be changed.
- Focus on monitoring, prioritization,
  segmentation, investigation, or review.

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