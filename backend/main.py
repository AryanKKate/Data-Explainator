from services.load_data import DataLoader
from graphs.analyst_graph import app
from pprint import pprint


df=DataLoader.load(
"data/customer_churn.csv"
)


result=app.invoke({

"data":df,

"profile":{},

"user_query":
"Predict customer churn based on features"})
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


# print("\n" + "="*70)
# print("TASK")
# print("="*70)

# print("Task Type:", result["task_type"])
# print("Target:", result["target_column"])

# # =====================================================
# # MODEL PLAN
# # =====================================================

# print("\n" + "="*70)
# print("MODEL PLAN")
# print("="*70)

# pprint(result["model_plan"])

# # =====================================================
# # TRAINING RESULTS
# # =====================================================

# print("\n" + "="*70)
# print("MODEL PERFORMANCE")
# print("="*70)

# best_model = result[
#     "training_results"
# ]["best_model"]

# print(
#     f"\nBest Model: {best_model}"
# )

# for model, metrics in result[
#     "training_results"
# ]["results"].items():

#     print("\n" + "-"*50)
#     print(model.upper())
#     print("-"*50)

#     for key, value in metrics.items():

#         if key != "top_features":

#             print(
#                 f"{key}: {value}"
#             )



print(
    result["explainability_report"]
)

print(
    result["insights"]
)

print(
    result["recommendations"]
)

print(
    result["executive_summary"]
)
