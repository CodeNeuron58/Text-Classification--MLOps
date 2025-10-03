import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='CodeNeuron58', repo_name='Text-Classification--MLOps', mlflow=True)

# Custom logger
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import get_logger
logger = get_logger("model_evaluation")


def load_data(test_data_path):
    logger.info(f"Loading test data from {test_data_path}")
    try:
        test_data = pd.read_csv(test_data_path)
        logger.info(f"Test data loaded successfully with shape {test_data.shape}")
        return test_data
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


def split_data(test_data):
    logger.info("Splitting test features and target...")
    try:
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        logger.info(f"Split completed. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Error splitting test data: {e}")
        raise


def load_model(model_path):
    logger.info(f"Loading model from {model_path}")
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


def save_metrics(metrics, output_path="reports/metrics.json"):
    logger.info(f"Saving metrics to {output_path}")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.info("Metrics saved successfully.")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise
    
def save_model_info(run_id: str, model_path: str, output_path: str):
    logger.info(f"Saving model info to {output_path}")
    try:
        model_info = {
            "run_id": run_id,
            "model_path": model_path
        }
        with open(output_path, "w") as file:
            json.dump(model_info, file, indent=4)
        logger.info("Model info saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model info: {e}")
        raise


def main():
    mlflow.set_experiment("DVC Pipeline")
    with mlflow.start_run() as run:
        test_data_path = "data/processed/test.csv"
        test_data = load_data(test_data_path)
        X_test, y_test = split_data(test_data)
        model_path = "models/model.pkl"
        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics)
        
        # Log the model type (e.g., Logistic Regression)
        model_type = type(model).__name__  # This will give you the model's class name, e.g., 'LogisticRegression'
        mlflow.log_param("model_type", model_type)
        
        # log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
            
        # Log model parameters to MLflow
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # save model info
        save_model_info(run.info.run_id, model_path, "reports/model_info.json")
        
        # Save and log the notebook
        mlflow.log_artifact(__file__)
        
        # log metrics file
        mlflow.log_artifact("reports/metrics.json")
        
        # log model file
        mlflow.log_artifact("models/model.pkl")
        
        # log test data file
        mlflow.log_artifact("data/processed/test.csv")
        
        mlflow.end_run()
        

if __name__ == "__main__":
    main()