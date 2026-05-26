from services.load_data import DataLoader
from graphs.analyst_graph import app

df=DataLoader.load(
    "data/sample.csv"
)

result=app.invoke({

    "data":df
})

print("\nCleaned Data:\n")

print(result["data"])

print("\nProfile:\n")

print(result["profile"])