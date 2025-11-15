# ============================================================== 
# FastAPI Application: Train, Fetch, and Predict using MLflow
# Single cached model instance for all endpoints
# ==============================================================

import os
import mlflow
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://34.63.106.86:5000/"  # MLflow tracking server
MODEL_NAME = "iris-random-forest"                   # Registered model name
RUN_NAME = "Random Forest Hyperparameter Search"    # MLflow run name

MODEL_DOWNLOAD_PATH = "downloaded_models"
MODEL_ARTIFACT_PATH = os.path.join(MODEL_DOWNLOAD_PATH, "random_forest_model")

LOCAL_ARTIFACT_DIR = "artifacts"
LOCAL_MODEL_PATH = os.path.join(LOCAL_ARTIFACT_DIR, "random_forest_model.pkl")

# Ensure directories exist
os.makedirs(MODEL_DOWNLOAD_PATH, exist_ok=True)
os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)

# --------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------
app = FastAPI(
    title="MLflow Model API",
    description="Train, fetch, and predict using MLflow-managed models with cached instance.",
    version="2.0.0",
)

# Global model cache
MODEL_CACHE = None

# --------------------------------------------------------------
# Request Schema
# --------------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --------------------------------------------------------------
# Utility: Prepare Data
# --------------------------------------------------------------
def prepare_data():
    try:
        data = pd.read_csv("./data.csv")
        data = pd.DataFrame(data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
        train, test = train_test_split(data, test_size=0.2, stratify=data["species"], random_state=42)
        feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        return train[feature_cols], train["species"], test[feature_cols], test["species"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {e}")

# --------------------------------------------------------------
# Utility: Train RandomForest Model
# --------------------------------------------------------------
def tune_random_forest(X_train, y_train, X_test, y_test):
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        rf_param_grid = {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [3, 5, 10],
            "class_weight": [None, "balanced"],
        }

        with mlflow.start_run(run_name=RUN_NAME):
            rf_model = RandomForestClassifier(random_state=42)
            rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)
            rf_grid.fit(X_train, y_train)

            best_model = rf_grid.best_estimator_
            mlflow.log_params(rf_grid.best_params_)
            mlflow.log_metric("best_cv_accuracy", rf_grid.best_score_)
            mlflow.log_metric("final_test_accuracy", rf_grid.score(X_test, y_test))
            mlflow.sklearn.log_model(best_model, "random_forest_model", registered_model_name=MODEL_NAME)

            # Save locally
            joblib.dump(best_model, LOCAL_MODEL_PATH)
            global MODEL_CACHE
            MODEL_CACHE = best_model  # Cache in memory
            return {
                "best_params": rf_grid.best_params_,
                "cv_accuracy": rf_grid.best_score_,
                "test_accuracy": rf_grid.score(X_test, y_test),
                "local_model_path": LOCAL_MODEL_PATH,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {e}")

# --------------------------------------------------------------
# Utility: Fetch Latest MLflow Model
# --------------------------------------------------------------
def fetch_latest_model():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(filter_string=f"name='{MODEL_NAME}'", order_by=["version_number DESC"], max_results=1)
        if not versions:
            raise HTTPException(status_code=404, detail="No registered versions found.")

        latest = versions[0]
        downloaded_path = mlflow.artifacts.download_artifacts(run_id=latest.run_id, artifact_path="random_forest_model", dst_path=MODEL_DOWNLOAD_PATH)
        return downloaded_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model: {e}")

# --------------------------------------------------------------
# Utility: Load Model into Cache
# --------------------------------------------------------------
def load_model_into_cache():
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE
    if os.path.exists(LOCAL_MODEL_PATH):
        MODEL_CACHE = joblib.load(LOCAL_MODEL_PATH)
        return MODEL_CACHE
    if os.path.exists(MODEL_ARTIFACT_PATH):
        MODEL_CACHE = mlflow.sklearn.load_model(MODEL_ARTIFACT_PATH)
        joblib.dump(MODEL_CACHE, LOCAL_MODEL_PATH)
        return MODEL_CACHE
    # Fetch from MLflow if missing
    downloaded_path = fetch_latest_model()
    if os.path.exists(MODEL_ARTIFACT_PATH):
        MODEL_CACHE = mlflow.sklearn.load_model(MODEL_ARTIFACT_PATH)
        joblib.dump(MODEL_CACHE, LOCAL_MODEL_PATH)
        return MODEL_CACHE
    raise HTTPException(status_code=404, detail="Model not found.")

# --------------------------------------------------------------
# FastAPI Endpoints
# --------------------------------------------------------------
@app.get("/", summary="Welcome")
def read_root():
    return {"message": "Welcome to the Iris Classifier API"}

@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "healthy"}

@app.post("/train", summary="Train Model")
def train_model():
    X_train, y_train, X_test, y_test = prepare_data()
    result = tune_random_forest(X_train, y_train, X_test, y_test)
    return {"status": "success", "details": result}

@app.get("/fetch", summary="Fetch Latest Model")
def fetch_model():
    path = fetch_latest_model()
    return {"status": "success", "download_path": path}

@app.post("/predict", summary="Predict Species")
def predict(input_data: IrisInput):
    model = load_model_into_cache()
    data = pd.DataFrame([[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]],
                        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    try:
        preds = model.predict(data)
        return {"status": "success", "prediction": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# --------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI app on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
