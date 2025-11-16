# ==============================================================
# Optimized FastAPI + MLflow Application
# --------------------------------------------------------------
# Features:
# - Background model loading from MLflow Model Registry
# - Caches model in memory + local file
# - JSON-safe structured logging
# - Readiness probe checks actual model availability
# ==============================================================

import os
import time
import json
import joblib
import asyncio
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import mlflow

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# --------------------------
# Tracing setup
# --------------------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# --------------------------
# Logging setup
# --------------------------
logger = logging.getLogger("iris-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# --------------------------
# Configuration
# --------------------------
MLFLOW_TRACKING_URI = "http://34.59.234.84:5000/"
MODEL_NAME = "iris-random-forest"
LOCAL_MODEL_DIR = "models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pkl")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Global model cache
MODEL_CACHE = None

# App state
app_state = {"is_alive": True, "is_ready": False}

# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(
    title="MLflow Optimized Iris API",
    description="Train, fetch, and predict using MLflow Model Registry.",
    version="3.0.0"
)

# --------------------------
# Payload Schema
# --------------------------
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --------------------------
# Utility Functions
# --------------------------

def load_latest_mlflow_model():
    """Load model from MLflow Registry and save locally."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        joblib.dump(model, LOCAL_MODEL_PATH)
        return model
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        raise

def load_model_cached():
    """Load model from memory, local disk, or MLflow."""
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE
    if os.path.exists(LOCAL_MODEL_PATH):
        MODEL_CACHE = joblib.load(LOCAL_MODEL_PATH)
        return MODEL_CACHE
    MODEL_CACHE = load_latest_mlflow_model()
    return MODEL_CACHE

def predict_with_cache_model(df: pd.DataFrame):
    """Make prediction using cached model."""
    if MODEL_CACHE is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        pred = MODEL_CACHE.predict(df)
        return {"prediction": pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

def prepare_data():
    """Prepare training data from CSV."""
    try:
        df = pd.read_csv("./data.csv")
        X_train, X_test, y_train, y_test = train_test_split(
            df[["sepal_length","sepal_width","petal_length","petal_width"]],
            df["species"],
            test_size=0.2,
            random_state=42,
            stratify=df["species"]
        )
        return X_train, y_train, X_test, y_test
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data: {e}")

def tune_random_forest(X_train, y_train, X_test, y_test):
    """Train RandomForest and register in MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini","entropy"],
            "max_depth": [None,5,10],
            "min_samples_split": [3,5,10],
            "class_weight": [None, "balanced"]
        }
        with mlflow.start_run(run_name="Random Forest Hyperparameter Search"):
            model = RandomForestClassifier(random_state=42)
            grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_accuracy", grid.best_score_)
            mlflow.log_metric("test_accuracy", grid.score(X_test, y_test))
            mlflow.sklearn.log_model(best, "model", registered_model_name=MODEL_NAME)
            joblib.dump(best, LOCAL_MODEL_PATH)
            global MODEL_CACHE
            MODEL_CACHE = best
            return {
                "best_params": grid.best_params_,
                "cv_accuracy": grid.best_score_,
                "test_accuracy": grid.score(X_test, y_test)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

# --------------------------
# Startup Event: Load model in background
# --------------------------
@app.on_event("startup")
async def startup_event():
    async def load_model_bg():
        global MODEL_CACHE
        try:
            MODEL_CACHE = load_model_cached()
            app_state["is_ready"] = True
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model on startup: {e}")
    asyncio.create_task(load_model_bg())

# --------------------------
# Probes
# --------------------------
@app.get("/live_check")
async def liveness_probe():
    return {"status": "alive"} if app_state["is_alive"] else Response(status_code=500)

@app.get("/ready_check")
async def readiness_probe():
    return {"status": "ready"} if MODEL_CACHE is not None else Response(status_code=503)

# --------------------------
# Middleware
# --------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = str(round((time.time()-start_time)*1000,2))
    return response

# --------------------------
# Exception Handler
# --------------------------
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event":"unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(status_code=500, content={"detail":"Internal Server Error", "trace_id": trace_id})

# --------------------------
# Endpoints
# --------------------------
@app.get("/")
def root():
    return {"message": "Optimized FastAPI + MLflow API"}

@app.get("/health")
def health():
    return {"status":"healthy"}

@app.post("/train")
def train():
    X_train, y_train, X_test, y_test = prepare_data()
    result = tune_random_forest(X_train, y_train, X_test, y_test)
    return {"status":"trained","details": result}

@app.get("/fetch")
def fetch_latest():
    model = load_latest_mlflow_model()
    return {"status":"fetched","local_path": LOCAL_MODEL_PATH}

@app.post("/predict")
async def predict(input: IrisInput, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id,"032x")
        df = pd.DataFrame([[input.sepal_length,input.sepal_width,input.petal_length,input.petal_width]],
                          columns=["sepal_length","sepal_width","petal_length","petal_width"])
        try:
            result = predict_with_cache_model(df)
            latency = round((time.time() - start_time)*1000,2)
            logger.info(json.dumps({
                "event":"prediction",
                "trace_id": trace_id,
                "input": df.to_dict(orient="records"),
                "result": result,
                "latency_ms": latency,
                "status":"success"
            }))
            return result
        except Exception as e:
            logger.exception(json.dumps({
                "event":"prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")

# --------------------------
# Run standalone
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
