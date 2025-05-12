import os
import random
import shutil
from data_test import test_dataset_normality
from path_grapper import get_all_data_pathnames

file_endings = {"rgb": ".png",
                "depth": "_depth.npy",
                "depth_mask": "_depth_mask.npy"}


def subset_full_dataset(amount_samples: int):
    """
    This will sample n amount of samples from the datapoints in ful_data
    """
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

    # test normality.
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


def main() -> None:
    """
    This main to to run the data subset maker on it's own
    """
    subset_full_dataset(200)


if __name__ == '__main__':
    main()
