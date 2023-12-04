import os
import pandas as pd

from vocal_patterns.params import DATA_TARGET


def get_data(test: bool = False) -> pd.DataFrame:
    if DATA_TARGET == "local":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        up_two_parents = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.dirname(up_two_parents)
        data_file_path = os.path.join(up_two_parents, "data")
        selected_file = "raw_data_test.csv" if test else "raw_data_train.csv"
        csv_path = os.path.join(data_file_path, selected_file)
        data = pd.read_csv(csv_path)

        # Apply the base path to the relative path in the CSV
        data["path"] = data["path"].apply(lambda x: os.path.join(base_path, x))

    # if DATA_TARGET == "mlflow":
    # pass

    return data


def upload_data_to_gcp(
    gcp_project: str, query: str, csv, data, bucket_name: str, blob_name: str
) -> pd.DataFrame:
    """We use this function if we need to upload data with filenames matching the CSV"""
    # determine if the data is already in GCP
    # set the file paths for each row
    # upload a CSV with the file paths
    pass


if __name__ == "__main__":
    get_data()
