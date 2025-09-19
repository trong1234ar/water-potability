import json
from mlflow.tracking import MlflowClient
import mlflow
import dagshub

# --- Setup
dagshub.init(repo_owner="trong1234ar", repo_name="water-potability", mlflow=True)
mlflow.set_experiment("DVC_Pipeline")
mlflow.set_tracking_uri("https://dagshub.com/trong1234ar/water-potability.mlflow")

# --- Load run info
with open("reports/run_info.json", "r") as f:
    run_info = json.load(f)

run_id = run_info["run_id"]

# IMPORTANT:
# In your previous code, run_info["model_name"] was actually the *artifact_path*
# you passed to mlflow.sklearn.log_model(..., artifact_path="Best Model")
artifact_path = run_info.get("artifact_path", run_info.get("model_name", "model"))

# Choose the *registered model name* (the name in the registry)
registered_model_name = run_info.get("model_name", "water_potability_model")

# --- Build the correct runs:/ URI (NO 'artifacts/' prefix)
model_uri = f"runs:/{run_id}/{artifact_path}"

client = MlflowClient()

# Create the registered model if it doesn't exist
try:
    client.create_registered_model(registered_model_name)
except Exception:
    # likely already exists
    pass

# Register a new version directly (avoids the failing register_model one-liner)
mv = client.create_model_version(
    name=registered_model_name,
    source=model_uri,
    run_id=run_id,
)

# Set an alias instead of using stages ("Staging"/"Production")
# Pick any alias naming you like; here we mimic "staging"
client.set_registered_model_alias(
    name=registered_model_name,
    alias="staging",
    version=mv.version
)

print(
    f"Registered '{registered_model_name}' version {mv.version} from {model_uri} "
    f"and set alias '@staging'."
)
