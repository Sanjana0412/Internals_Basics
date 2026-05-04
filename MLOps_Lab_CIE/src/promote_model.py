import mlflow
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
model_name = "payflow-processing-seconds-predictor"

# 1. Assign 'live' to Version 1
client.set_registered_model_alias(model_name, "live", "1")

# 2. Train Version 2 (Challenger)
df = pd.read_csv("data/new_data.csv")
X = df.drop("processing_seconds", axis=1)
y = df["processing_seconds"]

with mlflow.start_run(run_name="Challenger_Model"):
    model = RandomForestRegressor(random_state=99)
    model.fit(X, y)
    mae_v2 = mean_absolute_error(y, model.predict(X))
    mlflow.log_metric("mae", mae_v2)
    mv2 = mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

# 3. Compare (Simplified for lab: comparing v2 mae against v1 recorded mae)
with open("results/step3_s6.json", "r") as f:
    mae_v1 = json.load(f)["source_metric_value"]

action = "kept"
champion = 1
if mae_v2 < mae_v1:
    client.set_registered_model_alias(model_name, "live", "2")
    action = "promoted"
    champion = 2

output = {
    "registered_model_name": model_name,
    "alias_name": "live",
    "champion_version": champion,
    "challenger_version": 2,
    "action": action
}

with open("results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=4)