import os
import random
import shutil
import threading
from psutil import virtual_memory
from data_test import test_dataset_normality
from path_grapper import get_all_data_pathnames

file_endings = {"rgb": ".png",
                "depth": "_depth.npy",
                "depth_mask": "_depth_mask.npy"}


def subset_full_dataset(amount_samples: int) -> None:
    """
    This will sample n amount of samples from the datapoints in ful_data
    """
    # list all file endings.
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  "subset_data")

    print("getting data points")
    all_datapoints_folder_pathnames = get_all_data_pathnames()

    amount_differnt_main_folder = len(all_datapoints_folder_pathnames)
    sample_per_main_folder = amount_samples // amount_differnt_main_folder

    print("sampling data points")
    selected_data: list[list[str]] = [
        random.sample(main_folder, sample_per_main_folder) for
        main_folder in all_datapoints_folder_pathnames]

    print("flattening data points")
    flatten_selected_data = []

    for foldered_data in selected_data:
        flatten_selected_data += foldered_data

    print("running normality tests")
    # test normality.
    if virtual_memory().total > 16000000000:
        threads: list[threading.Thread] = []
        for folder in selected_data:
            threads.append(threading.Thread(target=test_dataset_normality,
                                            args=(folder, "sub folder",)))

        threads.append(threading.Thread(target=test_dataset_normality,
                                        args=(flatten_selected_data,
                                              "whole data",)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
    else:
        for folder in selected_data:
            test_dataset_normality(folder, "sub folder")

        test_dataset_normality(flatten_selected_data, "whole selected data")

    print("copying over data to subset data")
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
