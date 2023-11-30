import os
import numpy as np

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_REGION = os.environ["GCP_REGION"]
MODEL_TARGET = os.environ["MODEL_TARGET"]
DATA_TARGET = os.environ["DATA_TARGET"]
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT = os.environ["MLFLOW_EXPERIMENT"]
MLFLOW_MODEL_NAME = os.environ["MLFLOW_MODEL_NAME"]
PREFECT_FLOW_NAME = os.environ["PREFECT_FLOW_NAME"]

LOCAL_DATA_PATH = os.path.join(os.path.expanduser("~"), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH = os.path.join(
    os.path.expanduser("~"), ".lewagon", "mlops", "training_outputs"
)
