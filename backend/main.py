from services.load_data import DataLoader
from graphs.analyst_graph import app


df=DataLoader.load(
"data/sample1.csv"
)


result=app.invoke({

"data":df,

"profile":{},

"user_query":
"Predict customer churn and explain the important factors affecting churn"

})


# print(
# result["task_type"]
# )

# print(
# result["plan"]
# )

# print(
# result["engineered_data"].head()
# )

# print(

# result[
# "engineered_data"
# ][
# "churn"
# ]
# .value_counts()

# )

# print(
# result[
# "validation_report"
# ]
# )

# print(result["engineered_data"].describe())

print(
    result[
        "model_plan"
    ]
)