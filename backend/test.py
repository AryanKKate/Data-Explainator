from agents.schema_agent import SchemaAgent

columns=[

"customer_id",
"age",
"monthly_spend",
"last_login",
"churn"

]

print(
SchemaAgent.analyze(columns)
)