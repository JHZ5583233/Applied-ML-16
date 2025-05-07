import os
import random


def get_all_data_pathnames():
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  "full_data")

    datapoint_directories_main = []
    all_datapoint_folder_pathnames = []

    for folder in os.listdir(data_directory):
        datapoint_directories_main.append(os.path.join(data_directory,
                                                       folder))

    for main_folder in datapoint_directories_main:
        datapoint_directories = [main_folder]
        for _ in range(2):
            new_datapoint_directories = []

            for path in datapoint_directories:
                sub_folders = os.listdir(path)

                for folder in sub_folders:
                    new_datapoint_directories.append(os.path.join(path,
                                                                  folder))

            datapoint_directories = new_datapoint_directories
        all_datapoint_folder_pathnames.append(datapoint_directories)

    return all_datapoint_folder_pathnames

def subset_full_dataset(amount_samples: int):
    all_datapoints_folder_pathnames = get_all_data_pathnames()

    amount_differnt_main_folder = len(all_datapoints_folder_pathnames)
    sample_per_main_folder = amount_samples // amount_differnt_main_folder

    selected_data: list[list[str]] = [
        random.sample(main_folder, sample_per_main_folder) for
        main_folder in all_datapoints_folder_pathnames]

    # TODO test normality.

def main():
    test_dataset_normality


if __name__ == '__main__':
    main()
