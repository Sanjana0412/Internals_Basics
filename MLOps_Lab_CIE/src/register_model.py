import mlflow
import json

mlflow.set_tracking_uri("sqlite:///mlflow.db")
with open("results/step1_s1.json", "r") as f:
    step1 = json.load(f)

model_name = "payflow-processing-seconds-predictor"
run_id = step1["best_run_id"]
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(model_uri, model_name)

output = {
    "registered_model_name": model_name,
    "version": int(result.version),
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": step1["best_metric_value"]
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)