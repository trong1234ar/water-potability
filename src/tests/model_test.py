import unittest
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads .env file if it exists
except ImportError:
    pass  # dotenv not installed, continue without it

# Initialize DagsHub for experiment tracking
# THIS IS DUE TO GITHUB ACTION CAN ACCESS - LOCALLY WILL NOT NEED
dagshub_token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("DAGSHUB_TOKEN")
if dagshub_token:
    print("DAGSHUB_TOKEN is set")
else:
    print("DAGSHUB_TOKEN is not set")
# Try to use cached token if environment variable is not set
if not dagshub_token:
    try:
        # This will use the token from dagshub login command or add_app_token()
        dagshub.auth.add_app_token(dagshub_token) if dagshub_token else None
    except:
        pass

if not dagshub_token:
    raise ValueError("DAGSHUB_USER_TOKEN or DAGSHUB_TOKEN environment variable is not set")
# Set environment variables for MLflow authentication
os.environ["MLFLOW_TRACKING_TOKEN"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
# Configure DagsHub MLflow tracking
dagshub_url = "https://dagshub.com"
repo_owner = "trong1234ar"
repo_name = "water-potability"
# Initialize DagsHub connection
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
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

model_name = "Best Model"

def get_experiment_id(experiment_name="DVC_Pipeline"):
    """Get experiment ID by name"""
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            # If experiment doesn't exist, return default experiment ID
            return "0"
    except Exception as e:
        print(f"Error getting experiment ID: {e}")
        return "0"  # Default experiment ID

class TestModelLoading(unittest.TestCase):
    """Test model performance"""
    def test_model_in_staging(self):
        """Test model performance in staging"""
        client = MlflowClient()
        
        # Test 1: Check if registered model exists
        try:
            registered_models = client.search_registered_models()
            model_found = False
            for model in registered_models:
                if model.name == model_name:
                    model_found = True
                    # print(f"Found registered model: {model.name}")
                    break
            
            self.assertTrue(model_found, f"Registered model '{model_name}' not found")
            
        except Exception as e:
            print(f"Error searching registered models: {e}")
            self.fail(f"Failed to find registered model: {e}")
    
    # def test_model_versions(self):
    #     """Test model versions and aliases"""
    #     client = MlflowClient()
        
    #     try:
    #         # Get all versions of the model
    #         versions = client.get_latest_versions(model_name)
    #         print(f"Found {len(versions)} model versions")
    #         self.assertGreater(len(versions), 0, "No model versions found")
            
    #         # Check for staging alias
    #         try:
    #             staging_version = client.get_model_version_by_alias(model_name, "staging")
    #             print(f"Found model version {staging_version.version} with staging alias")
    #             self.assertIsNotNone(staging_version, "No model found with staging alias")
    #         except Exception as alias_error:
    #             print(f"No staging alias found: {alias_error}")
    #             # This is not necessarily a failure, just log it
                
    #     except Exception as e:
    #         print(f"Error checking model versions: {e}")
    #         self.fail(f"Failed to check model versions: {e}")
    
    # def test_experiment_runs(self):
    #     """Test experiment runs and metrics"""
    #     client = MlflowClient()
        
    #     # Get the correct experiment ID
    #     experiment_id = get_experiment_id("DVC_Pipeline")
    #     print(f"Using experiment ID: {experiment_id}")
        
    #     try:
    #         # Search runs in the experiment (this is supported by DagsHub)
    #         runs = client.search_runs(
    #             experiment_ids=[experiment_id],
    #             max_results=10
    #         )
            
    #         print(f"Found {len(runs)} runs in experiment")
    #         self.assertGreater(len(runs), 0, "No runs found in experiment")
            
    #         # Check if any run has good accuracy
    #         high_accuracy_runs = []
    #         for run in runs:
    #             if run.data.metrics.get('accuracy', 0) > 0.8:
    #                 high_accuracy_runs.append(run)
    #                 print(f"Run {run.info.run_id}: accuracy = {run.data.metrics.get('accuracy', 'N/A')}")
            
    #         if len(high_accuracy_runs) > 0:
    #             print(f"Found {len(high_accuracy_runs)} runs with accuracy > 0.8")
    #         else:
    #             print("No runs found with accuracy > 0.8")
                
    #     except Exception as e:
    #         print(f"Error searching runs: {e}")
    #         self.fail(f"Failed to search runs: {e}")
    
    # def test_model_performance(self):
    #     """Test model performance using search_logged_models"""
    #     client = MlflowClient()
    #     # Get the correct experiment ID
    #     experiment_id = get_experiment_id("DVC_Pipeline")
    #     print(f"Using experiment ID: {experiment_id}")
        
    #     # Search for models with good performance
    #     try:
    #         result = client.search_logged_models(
    #             experiment_ids=[experiment_id],
    #             filter_string="metrics.accuracy > 0.5",  # Lower threshold for testing
    #         )
            
    #         print(f"Found {len(result)} models with accuracy > 0.5")
    #         self.assertGreater(len(result), 0, "No models found with good performance")
            
    #         # Check if any model has accuracy > 0.85
    #         high_performance_models = client.search_logged_models(
    #             experiment_ids=[experiment_id],
    #             filter_string="metrics.accuracy > 0.85",
    #         )
            
    #         if len(high_performance_models) > 0:
    #             print(f"Found {len(high_performance_models)} high-performance models (accuracy > 0.85)")
    #         else:
    #             print("No models found with accuracy > 0.85")
                
    #     except Exception as e:
    #         print(f"Error searching logged models: {e}")
    #         self.fail(f"Failed to search logged models: {e}")
        

if __name__ == "__main__":
    unittest.main()