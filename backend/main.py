from services.load_data import DataLoader
from graphs.analyst_graph import app


df=DataLoader.load(
"data/sample1.csv"
)


result=app.invoke({

"data":df,

"profile":{},

"user_query":
"PRedict customer churn"

})
print(
    result[
        "semantic_schema"
    ]
)

print(
result["task_type"]
)

print(
result["plan"]
)

print(
result["engineered_data"].head()
)

target=(

    result[
        "target_column"
    ]

)

print(

    result[
        "engineered_data"
    ][target]

)



print(
result[
"validation_report"
]
)

print(result["engineered_data"].describe())

print(
    result[
        "model_plan"
    ]
)

print(
    result[
        "training_results"
    ]
)