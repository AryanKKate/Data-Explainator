from utils.llm import llm


class ExplainabilityAgent:

    @staticmethod
    def generate(

        target,
        schema,
        semantic_schema,
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

        top_features = result.get(
            "top_features",
            {}
        )

        prompt = f"""

You are a senior data analyst.

Target variable:

{target}

Model Used:

{best_model}

Model Metrics:

{result}

Feature Information:

{semantic_schema}

Insights:
{insights}

Top Features:

{top_features}

Recommendations:
{recommendations}

Generate:

1. Model performance summary

2. Key drivers

3. Business insights

4. Risks / limitations

5. Recommendations

Keep response concise and professional.

"""

        response = llm.invoke(
            prompt
        )

        return response.content