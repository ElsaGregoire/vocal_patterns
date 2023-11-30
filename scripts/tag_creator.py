import os
import csv

import pandas as pd
from sklearn.model_selection import train_test_split

# techniques = [
#     "belt",
#     "breathy",
#     "fast_forte",
#     "fast_piano",
#     "lip_trill",
#     "slow_forte",
#     "slow_piano",
#     "straight",
#     "vibrato",
#     "vocal_fry",
# ]


def generate_csv(base_dir, raw_data_folder_path, csv_file_path):
    exercises = [
        "arpeggios",
        "scales",
    ]
    with open(csv_file_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [
                "path",
                "exercise",
                "technique",
                "filename",
            ]
        )
        for dirpath, dirnames, filenames in os.walk(raw_data_folder_path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    exercise = next((e for e in exercises if e in filename), "other")
                    technique = dirpath.split("/")[-1]
                    full_path = os.path.join(str(dirpath), filename)
                    # Each user will store data in their own file system, so we remove the base_dir from the full path and publish the relative path to the csv
                    relative_path = os.path.relpath(full_path, base_dir)
                    csv_writer.writerow([relative_path, exercise, technique, filename])


def split_train_test(csv_file_path, train_file_path, test_file_path, test_size=0.2):
    data = pd.read_csv(csv_file_path)
    data_train, data_test = train_test_split(data, test_size=test_size)
    data_train.to_csv(train_file_path, mode="x", index=False)
    data_test.to_csv(test_file_path, mode="x", index=False)


def generate_train_test_csv(data_folder_path, base_dir, test_size=0.2):
    raw_data_folder_path = os.path.join(data_folder_path, "raw_data")
    csv_path = os.path.join(raw_data_folder_path, "raw_data.csv")
    train_file_path = os.path.join(data_folder_path, "raw_data_train.csv")
    test_file_path = os.path.join(data_folder_path, "raw_data_test.csv")

    generate_csv(base_dir, raw_data_folder_path, csv_path)
    split_train_test(csv_path, train_file_path, test_file_path, test_size=test_size)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    base_dir = os.path.dirname(parent_dir)
    relative_path = "data"
    data_folder_path = os.path.join(parent_dir, relative_path)

    generate_train_test_csv(data_folder_path, base_dir, test_size=0.2)

    print("CSV file has been generated in " + data_folder_path + ".")
