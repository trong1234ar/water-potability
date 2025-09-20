import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
import dagshub
import mlflow
import os
from mlflow.models import infer_signature

# Initialize DagsHub for experiment tracking
# Initialize DagsHub for experiment tracking
# dagshub.init(repo_owner='trong1234ar', repo_name='water-potability', mlflow=True)

# mlflow.set_experiment("DVC_Pipeline")
# THIS IS DUETO FOR GITHUB ACTION CAN ACCESS - LOCALLY WILL NOT NEED
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise ValueError("DAGSHUB_TOKEN environment variable is not set")
# Set environment variables for MLflow authentication
os.environ["MLFLOW_TRACKING_TOKEN"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
# Configure DagsHub MLflow tracking
dagshub_url = "https://dagshub.com"
repo_owner = "trong1234ar"
repo_name = "water-potability"
# Initialize DagsHub connection
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
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
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")

def evaluation_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]
        
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_param("Test_size",test_size)
        mlflow.log_param("n_estimators",n_estimators) 

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        
        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)
        
        # Log the model
        #mlflow.sklearn.log_model(model, model_name.replace(' ', '_'))

        metrics_dict = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")

def save_metrics(metrics: dict, metrics_path: str) -> None:
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}: {e}")

def main():
    try:
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"
        model_name = "Best Model"

        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)

        # Start MLflow run
        mlflow.autolog()

        with mlflow.start_run() as run:
            metrics = evaluation_model(model, X_test, y_test, model_name)
            save_metrics(metrics, metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path, "Best Model")  # Log model.pkl as Best_Model
            # mlflow.log_artifact(metrics_path)
            
            # Log the source code file
            mlflow.log_artifact(__file__)
            # signature = infer_signature(X_test,model.predict(X_test))

            # mlflow.sklearn.log_model(model,"Best Model",signature=signature)

            #Save run ID and model info to JSON File
            import os
            os.makedirs("reports", exist_ok=True)
            run_info = {'run_id': run.info.run_id, 'model_name': "Best Model"}
            reports_path = "reports/run_info.json"
            with open(reports_path, 'w') as file:
                json.dump(run_info, file, indent=4)

    except Exception as e:
        raise Exception(f"An Error occurred: {e}")

if __name__ == "__main__":
    main()