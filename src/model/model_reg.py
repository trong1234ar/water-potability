import json
from mlflow.tracking import MlflowClient
import mlflow
import dagshub
import os
# --- Setup
# Try to load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads .env file if it exists
except ImportError:
    pass  # dotenv not installed, continue without it

# Initialize DagsHub for experiment tracking
# Initialize DagsHub for experiment tracking
# dagshub.init(repo_owner='trong1234ar', repo_name='water-potability', mlflow=True)

# mlflow.set_experiment("DVC_Pipeline")
# THIS IS DUETO FOR GITHUB ACTION CAN ACCESS - LOCALLY WILL NOT NEED
dagshub_token = "6107d8a709e452caeccfaa8937ebed74cc0f1998"
if not dagshub_token:
    raise ValueError("DAGSHUB_TOKEN environment variable is not set")
# Set environment variables for MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = "trong1234ar"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
# Configure DagsHub MLflow tracking
dagshub_url = "https://dagshub.com"
repo_owner = "trong1234ar"
repo_name = "water-potability"
# Initialize Dagsub connection
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# Set experiment (this should work now with proper authentication)
try:
    mlflow.set_experiment("DVC_Pipeline")
except Exception as e:
    print(f"Warning: Could not set experiment 'DVC_Pipeline': {e}")
    # Create experiment if it doesn't exist
    try:
        mlflow.create_experiment("DVC_Pipeline")
        mlflow.set_experiment("DVC_Pipeline")
    except Exception as create_error:
        print(f"Error creating experiment: {create_error}")
        # Use default experiment as fallback
        print("Using default experiment")
# Set MLflow tracking URI
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
