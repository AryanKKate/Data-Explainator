from services.load_data import DataLoader
from graphs.analyst_graph import app


df=DataLoader.load(
"data/housing.csv"
)


result=app.invoke({

"data":df,

"profile":{},

"user_query":
"Devlop a regression model to predict the house prices. Preprocess data in any way to find necessary to obtaain the best results"
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