from utils.llm import llm


class ExecutiveSummaryAgent:

    @staticmethod
    def generate(

        target,
        task,
        training_results,
        insights,
        recommendations

    ):

        best_model = training_results[
            "best_model"
        ]

        result = training_results[
                "results"
            ][best_model]


        # top_features = result.get(
        #     "top_features",
        #     {}
        # )


        insight_list = insights.get(
            "insights",
            []
        )

        recommendation_list = recommendations.get(
            "recommendations",
            []
        )

        opportunity_list = recommendations.get(
            "opportunities",
            []
        )

        risk_list = recommendations.get(
            "risks",
            []
        )

        quick_wins = recommendations.get(
            "quick_wins",
            []
        )


        prompt = f"""

You are a senior analytics consultant preparing an executive report.

Your job is to summarize the analytical findings.

==================================================
TASK
==================================================

{task}

==================================================
TARGET
==================================================

{target}

==================================================
MODEL
==================================================

Model Used:
{best_model}

Metrics:
{result}

==================================================
INSIGHTS
==================================================

{insight_list}

==================================================
RISKS
==================================================

{risk_list}

==================================================
RECOMMENDATIONS
==================================================

{recommendation_list}

==================================================
OPPORTUNITIES
==================================================

{opportunity_list}

==================================================
QUICK WINS
==================================================

{quick_wins}

==================================================
STRICT RULES
==================================================

Use ONLY information explicitly provided above.

Do NOT create:

- thresholds
- customer groups
- segment descriptions
- monetary values
- percentages
- business rules
- domain assumptions
- causal explanations

Do NOT infer anything from:

- feature names
- correlations
- model importance

Do NOT create:

- new insights
- new recommendations
- new risks
- new opportunities

If information is not available,
write:

"Evidence not available."

Every statement in the report must originate from:

- Metrics
- Insights
- Risks
- Recommendations
- Opportunities
- Quick Wins

Do not introduce any new facts.

Do not recommend:

- feature engineering
- retraining
- model tuning
- collecting more data
- reviewing feature importance

unless explicitly present in Recommendations.

==================================================
OUTPUT FORMAT
==================================================

## Model Performance Summary

Briefly summarize model performance using only the provided metrics.

## Key Drivers

List the most important drivers mentioned in the insights.

## Business Insights

Summarize only the business insights provided.

## Risks / Limitations

Use ONLY the provided risks.

If no risks exist, write:

"No major analytical risks identified."

## Recommendations

List only the provided recommendations.

## Opportunities

List only the provided opportunities.

## Quick Wins

List only the provided quick wins.

Keep the report concise, professional, and executive-friendly.

"""

        response = llm.invoke(
            prompt
        )

        return response.content