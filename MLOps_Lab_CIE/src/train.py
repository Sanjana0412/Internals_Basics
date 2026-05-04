import pandas as pd
import numpy as np
import mlflow
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("payflow-processing-seconds")
df = pd.read_csv("data/training_data.csv")
X = df.drop("processing_seconds", axis=1)
y = df["processing_seconds"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results_list = []
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

best_mae = float('inf')
best_model_name = ""
best_run_id = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("domain", "fintech___payments")
        mlflow.sklearn.log_model(model, "model")
        
        results_list.append({"name": name, "mae": round(mae, 4), "rmse": round(rmse, 4)})
        
        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_run_id = mlflow.active_run().info.run_id
            # Save local copy for Docker Task
            joblib.dump(model, "models/model.pkl")

# Save JSON
output = {
    "experiment_name": "payflow-processing-seconds",
    "models": results_list,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": round(best_mae, 4),
    "best_run_id": best_run_id
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)