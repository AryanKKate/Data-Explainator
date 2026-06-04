from services.load_data import DataLoader
from graphs.analyst_graph import app
from pprint import pprint


df=DataLoader.load(
"data/housing.csv"
)


result=app.invoke({

"data":df,

"profile":{},

"user_query":
"Devlop a regression model to predict the house prices. Preprocess data in any way to find necessary to obtaain the best results"
})
# print(
#     result[
#         "semantic_schema"
#     ]
# )

# print(
# result["task_type"]
# )

# print(
# result["plan"]
# )

# print(
# result["engineered_data"].head()
# )

# target=(

#     result[
#         "target_column"
#     ]

# )

# print(

#     result[
#         "engineered_data"
#     ][target]

# )



# print(
# result[
# "validation_report"
# ]
# )

# print(result["engineered_data"].describe())

# print(
#     result[
#         "model_plan"
#     ]
# )

# print(
#     result[
#         "training_results"
#     ]
# )


print("\n" + "="*70)
print("TASK")
print("="*70)

print("Task Type:", result["task_type"])
print("Target:", result["target_column"])

# =====================================================
# MODEL PLAN
# =====================================================

print("\n" + "="*70)
print("MODEL PLAN")
print("="*70)

pprint(result["model_plan"])

# =====================================================
# TRAINING RESULTS
# =====================================================

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)

best_model = result[
    "training_results"
]["best_model"]

print(
    f"\nBest Model: {best_model}"
)

for model, metrics in result[
    "training_results"
]["results"].items():

    print("\n" + "-"*50)
    print(model.upper())
    print("-"*50)

    for key, value in metrics.items():

        if key != "top_features":

            print(
                f"{key}: {value}"
            )

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

print("\n" + "="*70)
print("TOP FEATURES")
print("="*70)

best_metrics = result[
    "training_results"
]["results"][best_model]

if "top_features" in best_metrics:

    for feature, score in best_metrics[
        "top_features"
    ].items():

        print(
            f"{feature}: {score}"
        )

# =====================================================
# INSIGHTS
# =====================================================

if "insights" in result:

    print("\n" + "="*70)
    print("INSIGHTS")
    print("="*70)

    insights = result["insights"]

    if isinstance(insights, dict):

        for item in insights.get(
            "insights",
            []
        ):

            print(f"• {item}")

    else:

        print(insights)

# =====================================================
# RECOMMENDATIONS
# =====================================================

if "recommendations" in result:

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    recommendations = result[
        "recommendations"
    ]

    if isinstance(
        recommendations,
        dict
    ):

        for rec in recommendations.get(
            "recommendations",
            []
        ):

            print(f"✓ {rec}")

    else:

        print(recommendations)

# =====================================================
# EXPLANATION
# =====================================================

if "explanation" in result:

    print("\n" + "="*70)
    print("EXECUTIVE SUMMARY")
    print("="*70)

    print(
        result["explanation"]
    )

