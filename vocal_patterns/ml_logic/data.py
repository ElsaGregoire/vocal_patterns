from pathlib import Path
import pandas as pd


def upload_data_to_gcp(
    gcp_project: str, query: str, csv, data, bucket_name: str, blob_name: str
) -> pd.DataFrame:
    """We use this function if we need to upload data with filenames matching the CSV"""
    # determine if the data is already in GCP
    # set the file paths for each row
    # upload a CSV with the file paths
    pass
