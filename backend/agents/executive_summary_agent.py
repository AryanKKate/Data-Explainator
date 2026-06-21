from utils.llm import llm


class ExecutiveSummaryAgent:

    @staticmethod
    def generate(

        target,
        task,
        training_results,
        insights,
        recommendations,
        explainability_report,
        semantic_schema

    ):

        best_model = training_results[
            "best_model"
        ]

        result = training_results[
                "results"
            ][best_model]
            
        top_drivers = explainability_report.get(
        "top_drivers",
        []
    )

        importance = explainability_report.get(
            "global_importance_percentages",
            {}
        )
        # top_features = result.get(
        #     "top_features",
        #     {}
        # )

        prompt = f"""

You are a senior data analyst.

Target variable:

{target}

TOP DRIVERS

{top_drivers}

IMPORTANCE

{importance}

Model Used:

{best_model}

Model Metrics:

{result}

Feature Information:

{semantic_schema}

Insights:
{insights}


Recommendations:
{recommendations}
IMPORTANT RULES

Use ONLY information provided.

Never invent:

- thresholds
- customer segments
- monetary values
- business rules
- domain assumptions

If a threshold is not explicitly present
in Insights or Recommendations,
do not create one.

If evidence is unavailable,
say:

"Evidence not available."

Do not create explanations from
feature names alone.

Do not infer causation from
correlation.

Do not recommend:

- model tuning
- feature engineering
- retraining
- reviewing feature importance
- collecting more data

unless explicitly present in recommendations.

Every recommendation must originate from:
Recommendations section.

Every business insight must originate from:
Insights section.

Create a report using ONLY the supplied:

- Metrics
- Insights
- Recommendations

Do not create new insights.
Do not create new recommendations.
Do not create new risks.

Only organize and summarize
the provided information.

Keep response concise and professional.

"""

        response = llm.invoke(
            prompt
        )

        return response.content