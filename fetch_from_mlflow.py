import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# --------------------------
# Config
# --------------------------
# Use env var, fallback to local MLflow server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
MODEL_NAME = "iris-random-forest"
DOWNLOAD_DIR = "downloaded_models"
LOCAL_MODEL_PATH = os.path.join(DOWNLOAD_DIR, "model.pkl")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def fetch(client, name):
    """Fetch the latest registered model version from the MLflow Model Registry."""
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
        print(
            f"[INFO] Latest Version: v{latest.version}, "
            f"Stage: {latest.current_stage}, Run ID: {latest.run_id}"
        )
        return latest

    except Exception as e:
        print(f"[ERROR] While fetching metadata: {e}")
        return None

def load_model(version):
    """Load model directly from MLflow Model Registry."""
    if not version:
        print("[ERROR] Cannot load – model version metadata missing.")
        return None

    try:
        model_uri = f"models:/{MODEL_NAME}/{version.version}"
        print(f"[INFO] Loading model from Registry URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("[INFO] Model loaded successfully.")
        return model

    except MlflowException as e:
        print(f"[MLflow ERROR] {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

def download_artifacts(version):
    """Download underlying model artifacts."""
    if not version:
        print("[ERROR] Cannot download artifacts – version missing.")
        return None

    try:
        artifact_uri = f"models:/{MODEL_NAME}/{version.version}"
        print(f"[INFO] Downloading artifacts from: {artifact_uri}")

        downloaded_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=DOWNLOAD_DIR
        )

        print(f"[INFO] Artifacts downloaded to: {downloaded_path}")
        return downloaded_path

    except Exception as e:
        print(f"[ERROR] Failed to download artifacts: {e}")
        return None

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"[INFO] MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

    client = MlflowClient()
    version = fetch(client, MODEL_NAME)

    if version:
        load_model(version)
        download_artifacts(version)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[CRITICAL] {e}")
        sys.exit(1)
