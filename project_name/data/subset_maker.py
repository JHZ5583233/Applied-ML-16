import os
import random
import shutil
from data_test import test_dataset_normality

file_endings = {"rgb": ".png",
                "depth": "_depth.npy",
                "depth_mask": "_depth_mask.npy"}


def get_all_data_pathnames() -> list[list[str]]:
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  "full_data")

    datapoint_directories_main = []
    all_datapoint_folder_pathnames = []

    for folder in os.listdir(data_directory):
        datapoint_directories_main.append(os.path.join(data_directory,
                                                       folder))

    for main_folder in datapoint_directories_main:
        datapoint_directories: list[str] = [main_folder]
        for _ in range(3):
            new_datapoint_directories = []

            for path in datapoint_directories:
                sub_folders = os.listdir(path)

                for folder in sub_folders:
                    new_datapoint_directories.append(os.path.join(path,
                                                                  folder))

            # get rid of all duplicates
            datapoint_directories = new_datapoint_directories
        # get rid of file extension and type description.
        for ending in file_endings.values():
            datapoint_directories = [_.removesuffix(ending)
                                     for _ in datapoint_directories]

        all_datapoint_folder_pathnames.append(list(set(datapoint_directories)))

    return all_datapoint_folder_pathnames


def get_train_data_folders() -> list[str]:
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  "subset_data")

    return os.listdir(data_directory)


def subset_full_dataset(amount_samples: int):
    # list all file endings.
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  "subset_data")

    all_datapoints_folder_pathnames = get_all_data_pathnames()

    amount_differnt_main_folder = len(all_datapoints_folder_pathnames)
    sample_per_main_folder = amount_samples // amount_differnt_main_folder

    selected_data: list[list[str]] = [
        random.sample(main_folder, sample_per_main_folder) for
        main_folder in all_datapoints_folder_pathnames]
    flatten_selected_data = []

    for foldered_data in selected_data:
        flatten_selected_data += foldered_data

    # TODO test normality.
    for folder in selected_data:
        test_dataset_normality(folder)

    test_dataset_normality(flatten_selected_data)

    # copy over the subsetted data in their own folders.
    for index, data in enumerate(flatten_selected_data):
        data_point_folder = os.path.join(data_directory, str(index))
        try:
            os.mkdir(data_point_folder)
            print(f"made data point {index}")
        except FileExistsError:
            os.rmdir(data_point_folder)
            os.mkdir(data_point_folder)

        for endings in file_endings.values():
            shutil.copy(data + endings, data_point_folder)


def main():
    subset_full_dataset(200)


if __name__ == '__main__':
    main()
