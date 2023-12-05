import os
import csv

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
                "filename",
            ]
        )
        for dirpath, dirnames, filenames in os.walk(raw_data_folder_path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    exercise = dirpath.split("/")[-1]
                    full_path = os.path.join(str(dirpath), filename)
                    # Each user will store data in their own file system, so we remove the base_dir from the full path and publish the relative path to the csv
                    relative_path = os.path.relpath(full_path, base_dir)
                    csv_writer.writerow([relative_path, exercise, filename])


def generate(data_folder_path, base_dir):
    raw_data_folder_path = os.path.join(data_folder_path, "manual_test_data")
    csv_path = os.path.join(raw_data_folder_path, "manual_test_data.csv")

    generate_csv(base_dir, raw_data_folder_path, csv_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    base_dir = os.path.dirname(parent_dir)
    relative_path = "data"
    data_folder_path = os.path.join(parent_dir, relative_path)

    generate(data_folder_path, base_dir)

    print("CSV file has been generated in " + data_folder_path + ".")
