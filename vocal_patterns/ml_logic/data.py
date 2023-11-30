import os
import pandas as pd

from vocal_patterns.params import MODEL_TARGET


def get_training_data():
    if MODEL_TARGET == "local":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        parent_dir = os.path.dirname(parent_dir)
        relative_path = "data"
        data_file_path = os.path.join(parent_dir, relative_path)
        csv_path = os.path.join(data_file_path, "dataset_tags.csv")
        print(csv_path)

        data = pd.read_csv(csv_path)

    if MODEL_TARGET == "mlflow":
        pass
    return data


def upload_data_to_gcp(
    gcp_project: str, query: str, csv, data, bucket_name: str, blob_name: str
) -> pd.DataFrame:
    """We use this function if we need to upload data with filenames matching the CSV"""
    # determine if the data is already in GCP
    # set the file paths for each row
    # upload a CSV with the file paths
    pass
