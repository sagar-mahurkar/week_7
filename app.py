# ==============================================================
# Optimized FastAPI + MLflow Application
# --------------------------------------------------------------
# - Uses MLflow Model Registry instead of slow artifact download
# - Deletes ALL slow mlflow.artifacts.download_artifacts usage
# - Loads model via: models:/<model_name>/latest
# - Caches model in memory + local file
# - Much faster and avoids MLflow 500 errors
# ==============================================================

import os
import mlflow
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://34.173.23.187:5000/"   # MLflow tracking server
MODEL_NAME = "iris-random-forest"                    # Registered model name
RUN_NAME = "Random Forest Hyperparameter Search"     # Run name

LOCAL_MODEL_DIR = "models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pkl")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Global single model cache
MODEL_CACHE = None

# --------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------
app = FastAPI(
    title="MLflow Optimized Iris API",
    description="Train, fetch, and predict using MLflow Model Registry (fast).",
    version="3.0.0",
)

# --------------------------------------------------------------
# Payload Schema
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
        df = pd.read_csv("./data.csv")
        df = pd.DataFrame(df, columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

        X_train, X_test, y_train, y_test = train_test_split(
            df[["sepal_length", "sepal_width", "petal_length", "petal_width"]],
            df["species"],
            test_size=0.2,
            random_state=42,
            stratify=df["species"],
        )
        return X_train, y_train, X_test, y_test
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {e}")

# --------------------------------------------------------------
# Utility: Train Model + Register to MLflow
# --------------------------------------------------------------
def tune_random_forest(X_train, y_train, X_test, y_test):
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        param_grid = {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [3, 5, 10],
            "class_weight": [None, "balanced"],
        }

        with mlflow.start_run(run_name=RUN_NAME):
            base_model = RandomForestClassifier(random_state=42)
            grid = GridSearchCV(base_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
            grid.fit(X_train, y_train)

            best = grid.best_estimator_

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("best_cv_accuracy", grid.best_score_)
            mlflow.log_metric("test_accuracy", grid.score(X_test, y_test))

            # Register model into MLflow Model Registry
            mlflow.sklearn.log_model(best, "model", registered_model_name=MODEL_NAME)

            # Save locally
            joblib.dump(best, LOCAL_MODEL_PATH)

            global MODEL_CACHE
            MODEL_CACHE = best

            return {
                "best_params": grid.best_params_,
                "cv_accuracy": grid.best_score_,
                "test_accuracy": grid.score(X_test, y_test),
                "local_model_path": LOCAL_MODEL_PATH,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {e}")

# --------------------------------------------------------------
# Utility: Load Latest Model from MLflow Registry
# --------------------------------------------------------------
def load_latest_mlflow_model():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)

        joblib.dump(model, LOCAL_MODEL_PATH)   # Save local cache
        return model

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model from MLflow Registry: {e}")

# --------------------------------------------------------------
# Utility: Central Model Loader (cached)
# --------------------------------------------------------------
def load_model_cached():
    global MODEL_CACHE

    if MODEL_CACHE is not None:
        return MODEL_CACHE

    if os.path.exists(LOCAL_MODEL_PATH):
        MODEL_CACHE = joblib.load(LOCAL_MODEL_PATH)
        return MODEL_CACHE

    # Load from MLflow Registry
    MODEL_CACHE = load_latest_mlflow_model()
    return MODEL_CACHE

# --------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Optimized FastAPI + MLflow Model Registry API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/train")
def train():
    X_train, y_train, X_test, y_test = prepare_data()
    result = tune_random_forest(X_train, y_train, X_test, y_test)
    return {"status": "trained", "details": result}

@app.get("/fetch")
def fetch_latest():
    model = load_latest_mlflow_model()
    return {"status": "fetched", "local_path": LOCAL_MODEL_PATH}

@app.post("/predict")
def predict(data: IrisInput):
    model = load_model_cached()
    df = pd.DataFrame([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]],
                      columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    try:
        pred = model.predict(df)
        return {"prediction": pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

# --------------------------------------------------------------
# Standalone app run
# --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
