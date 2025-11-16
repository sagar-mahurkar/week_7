import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

URI = "http://34.59.234.84:5000/"
NAME = "iris-random-float"
SAVE_LOCAL = "downloaded_models/model.pkl"

os.makedirs("downloaded_models", exist_ok=True)


def fetch(client, name):
    """
    Fetch the latest registered model version (fast metadata only).
    """
    print(f"Searching for latest registered model: {name}")

    try:
        versions = client.search_model_versions(
            filter_string=f"name='{name}'",
            order_by=["version_number DESC"],
            max_results=1
        )
        
        if not versions:
            print(f"[ERROR] No versions found for model '{name}'.")
            return None

        latest = versions[0]
        print(f"[INFO] Latest Version: v{latest.version}, Stage: {latest.current_stage}, Run ID: {latest.run_id}")
        return latest

    except Exception as e:
        print(f"[ERROR] While fetching model metadata: {e}")
        return None


def load_model(version):
    """
    Load model directly from MLflow Model Registry instead of using artifact downloads.
    This avoids the slow /mlflow-artifacts endpoint entirely.
    """
    if not version:
        print("[ERROR] Cannot load model — version metadata missing.")
        return None

    try:
        model_uri = f"models:/{NAME}/{version.version}"
        print(f"[INFO] Loading model from Registry URI: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)

        # Optional: Save locally
        mlflow.pyfunc.save_model(model_uri=model_uri, path="downloaded_models")

        print(f"[INFO] Model loaded and saved locally to: {SAVE_LOCAL}")
        return model

    except MlflowException as e:
        print(f"[MLflow ERROR] {e}")
        return None

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


def main():
    mlflow.set_tracking_uri(URI)
    print(f"[INFO] MLflow tracking URI set to: {URI}")

    client = MlflowClient()

    version = fetch(client, NAME)

    if version:
        load_model(version)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[CRITICAL] {e}")
        sys.exit(1)
