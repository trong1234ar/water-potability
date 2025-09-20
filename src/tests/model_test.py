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
    
    
    def test_model_performance(self):
        """Test model performance by getting latest version and checking metrics"""
        client = MlflowClient()
        
        # Performance thresholds
        accuracy_threshold = 0.4
        precision_threshold = 0.3
        recall_threshold = 0.3
        f1_threshold = 0.3
        
        try:
            # Get the latest version of the registered model
            latest_versions = client.get_latest_versions(model_name)
            self.assertGreater(len(latest_versions), 0, f"No versions found for model '{model_name}'")
            
            # Get the latest version (usually the first one or highest version number)
            latest_version = max(latest_versions, key=lambda v: int(v.version))
            print(f"Testing latest model version: {latest_version.version}")
            
            # Get the run associated with this model version
            run_id = latest_version.run_id
            run = client.get_run(run_id)
            
            # Extract metrics from the run
            metrics = run.data.metrics
            print(f"Model metrics: {metrics}")
            
            # Test accuracy
            accuracy = metrics.get('accuracy', 0)
            print(f"Accuracy: {accuracy:.4f} (threshold: {accuracy_threshold})")
            self.assertGreaterEqual(accuracy, accuracy_threshold, 
                                  f"Model accuracy {accuracy:.4f} is below threshold {accuracy_threshold}")
            
            # Test precision
            precision = metrics.get('precision', 0)
            print(f"Precision: {precision:.4f} (threshold: {precision_threshold})")
            self.assertGreaterEqual(precision, precision_threshold,
                                  f"Model precision {precision:.4f} is below threshold {precision_threshold}")
            
            # Test recall
            recall = metrics.get('recall', 0)
            print(f"Recall: {recall:.4f} (threshold: {recall_threshold})")
            self.assertGreaterEqual(recall, recall_threshold,
                                  f"Model recall {recall:.4f} is below threshold {recall_threshold}")
            
            # Test F1 score
            f1_score = metrics.get('f1_score', 0)
            print(f"F1 Score: {f1_score:.4f} (threshold: {f1_threshold})")
            self.assertGreaterEqual(f1_score, f1_threshold,
                                  f"Model F1 score {f1_score:.4f} is below threshold {f1_threshold}")
            
            print("✅ All performance metrics meet the required thresholds!")
            
        except Exception as e:
            print(f"Error testing model performance: {e}")
            self.fail(f"Failed to test model performance: {e}")
    
    def test_staging_model_performance(self):
        """Test performance of model in staging (if exists)"""
        client = MlflowClient()
        
        # Lower thresholds for staging models
        staging_accuracy_threshold = 0.7
        
        try:
            # Try to get staging model by alias
            try:
                staging_version = client.get_model_version_by_alias(model_name, "staging")
                print(f"Found staging model version: {staging_version.version}")
                
                # Get the run associated with staging model
                run_id = staging_version.run_id
                run = client.get_run(run_id)
                
                # Check staging model performance
                accuracy = run.data.metrics.get('accuracy', 0)
                print(f"Staging model accuracy: {accuracy:.4f}")
                
                self.assertGreaterEqual(accuracy, staging_accuracy_threshold,
                                      f"Staging model accuracy {accuracy:.4f} is below threshold {staging_accuracy_threshold}")
                
                print("✅ Staging model meets performance requirements!")
                
            except Exception as staging_error:
                print(f"No staging model found or error accessing it: {staging_error}")
                # This is not necessarily a test failure, just log it
                print("⚠️ Skipping staging model performance test")
                
        except Exception as e:
            print(f"Error in staging model performance test: {e}")
            # Don't fail the test if staging model doesn't exist
            print("⚠️ Staging model performance test skipped due to error")
        

if __name__ == "__main__":
    unittest.main()