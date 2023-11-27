import os
import csv

exercises = [
    "arpeggios",
    "scales",
]

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


def get_relative_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, relative_path)

    return full_path


def generate_csv(root_path, csv_file_path):
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
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith(".wav"):
                    exercise = next((e for e in exercises if e in filename), "Other")
                    technique = dirpath.split("/")[-1]
                    full_path = os.path.join(str(dirpath), filename)
                    csv_writer.writerow([full_path, exercise, technique, filename])


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    relative_path = "vocal_patterns/data"
    data_file_path = os.path.join(parent_dir, relative_path)
    csv_path = os.path.join(data_file_path, "dataset_tags.csv")
    print(csv_path)

    generate_csv(data_file_path, csv_path)
    print("CSV file has been generated at " + csv_path + ".")
