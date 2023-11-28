from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from vocal_patterns.params import MODEL_TARGET


def get_data():
    if MODEL_TARGET == "local":
        download_path = "../vocal_patterns/data/dataset_tags.csv"
        data = pd.read_csv(download_path)

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
