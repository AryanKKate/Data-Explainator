from services.load_data import DataLoader
from graphs.analyst_graph import app


df=DataLoader.load(
"data/sample.csv"
)


result=app.invoke({

"data":df,

"profile":{},

"user_query":
"Predict gadget y sales for the next quarter"

})


print(
result["task_type"]
)

print(
result["plan"]
)

print(
result["engineered_data"].head()
)