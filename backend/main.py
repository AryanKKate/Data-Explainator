from services.load_data import DataLoader
from graphs.analyst_graph import app


df=DataLoader.load(
"data/sample.csv"
)


result=app.invoke({

"data":df,

"profile":{},

"user_query":
"Predict inventory status using all available product, warehouse, supplier, pricing, quantity, and restocking information. Automatically clean inconsistent values, detect semantic column meanings, engineer useful features, select the best ML model adaptively, and explain the most important factors affecting stock status."

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