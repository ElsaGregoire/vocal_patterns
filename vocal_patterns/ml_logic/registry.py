import glob
import json
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

import mlflow
from mlflow.tracking import MlflowClient

from vocal_patterns.params import (
    LOCAL_REGISTRY_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_TARGET,
)


def save_results(params: dict, metrics: dict) -> None:
    if MODEL_TARGET == "mlflow":
        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)
        print("✅ Results saved on MLflow")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(
            LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle"
        )
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


# def save_history(history: keras.callbacks.History):
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     if history is not None:
#         history_path = os.path.join(
#             LOCAL_REGISTRY_PATH, "history", timestamp + ".pickle"
#         )
#         with open(history_path, "wb") as file:
#             pickle.dump(history, file)

#     print("✅ History saved locally")


# def load_history():
#     local_history_directory = os.path.join(LOCAL_REGISTRY_PATH, "history")
#     local_history_paths = glob.glob(f"{local_history_directory}/*")

#     if not local_history_paths:
#         return None

#     most_recent_history_path_on_disk = sorted(local_history_paths)[-1]
#     print("most_recent_history_path_on_disk", most_recent_history_path_on_disk)
#     print(Fore.BLUE + f"\nLoad latest history from disk..." + Style.RESET_ALL)
#     with open(most_recent_history_path_on_disk, "rb") as file:
#         history = pickle.load(file)
#     print("✅ History loaded from local disk")
#     return history


def save_model(model: keras.Model = None, augmentations=None) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.augmentations = augmentations
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "mlflow":
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )
        with open("augmentations.json", "w") as f:
            json.dump(augmentations, f)
        mlflow.log_artifact("augmentations.json")
        print("✅ Model saved to MLflow")
        return None
    return None


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(
            Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL
        )

        # Get the latest model version name by the timestamp on disk

        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print("most_recent_model_path_on_disk", most_recent_model_path_on_disk)
        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
        latest_model = keras.models.load_model(most_recent_model_path_on_disk)
        print("✅ Model loaded from local disk")
        latest_model.timestamp = most_recent_model_path_on_disk.split("/")[-1].split(
            "."
        )[0]
        params_path = os.path.join(
            LOCAL_REGISTRY_PATH, "params", latest_model.timestamp + ".pickle"
        )
        with open(params_path, "rb") as file:
            latest_model.params = pickle.load(file)
        latest_model.augmentations = latest_model.params["data_augmentations"]
        return latest_model

    elif MODEL_TARGET == "mlflow":
        print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

        # Load model from MLflow
        model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            model_versions = client.get_latest_versions(
                name=MLFLOW_MODEL_NAME, stages=[stage]
            )
            model_uri = model_versions[0].source

            assert model_uri is not None
        except:
            print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")

            return None

        model = mlflow.tensorflow.load_model(model_uri=model_uri)

        ### ARTIFACTS
        run_id = os.path.basename(os.path.dirname(os.path.dirname(model_uri)))
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="augmentations.json"
        )
        with open(local_path, "r") as f:
            augmentations = json.load(f)
        model.augmentations = augmentations
        ######
        model.timestamp = len(model_versions)
        print("✅ Model loaded from MLflow")
        return model
    else:
        return None


def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

    if not version:
        print(
            f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}"
        )
        return None

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True,
    )

    print(
        f"✅ Model {MLFLOW_MODEL_NAME} (version {version[0].version}) transitioned from {current_stage} to {new_stage}"
    )

    return None


def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    print("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)

    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results

    return wrapper
