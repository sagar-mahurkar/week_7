# train.py
import os
import mlflow
import joblib
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://34.59.234.84:5000/"
MODEL_NAME = "iris-random-forest"
RUN_NAME = "Random Forest Hyperparameter Search"

LOCAL_MODEL_DIR = "artifacts"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "random_forest_model.pkl")

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


# --------------------------------------------------------------
# Data Preparation
# --------------------------------------------------------------
def prepare_data():
    """Load and split the Iris dataset."""
    try:
        data = pd.read_csv("./data.csv")
        data = data[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]]

        train, test = train_test_split(
            data, test_size=0.2, stratify=data["species"], random_state=42
        )

        X_train = train.drop(columns=["species"])
        y_train = train["species"]
        X_test = test.drop(columns=["species"])
        y_test = test["species"]

        return X_train, y_train, X_test, y_test

    except Exception as e:
        raise RuntimeError(f"Error preparing data: {e}")


# --------------------------------------------------------------
# Train + Hyperparameter Tune + Log to MLflow
# --------------------------------------------------------------
def tune_random_forest(X_train, y_train, X_test, y_test):
    """Train and tune a RandomForest model, log to MLflow registry."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        param_grid = {
            "n_estimators": [50, 100],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        }

        with mlflow.start_run(run_name=RUN_NAME):
            model = RandomForestClassifier(random_state=42)

            grid = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            # Log params & metrics
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_accuracy", grid.best_score_)
            mlflow.log_metric("test_accuracy", best_model.score(X_test, y_test))

            # Log model to MLflow Registry (fast — NO downloads)
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )

            # Save locally for FastAPI inference cache
            joblib.dump(best_model, LOCAL_MODEL_PATH)

            return {
                "best_params": grid.best_params_,
                "cv_accuracy": grid.best_score_,
                "test_accuracy": best_model.score(X_test, y_test),
                "local_model_path": LOCAL_MODEL_PATH,
            }

    except Exception as e:
        raise RuntimeError(f"Error during training: {e}")


# --------------------------------------------------------------
# Main Entry
# --------------------------------------------------------------
def main():
    print("\n🚀 Starting Model Training...")
    X_train, y_train, X_test, y_test = prepare_data()
    result = tune_random_forest(X_train, y_train, X_test, y_test)
    print("\n✅ Training Complete!")
    print(result)


if __name__ == "__main__":
    main()

        
