import os
import random
import shutil
import threading
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from psutil import virtual_memory
from shutil import rmtree
from data_test import (test_dataset_normality,
                       threaded_make_data_array,
                       multithread_data_test_output)
from path_grapper import get_all_data_path_names


def subset_full_dataset(amount_samples: int, full_data_folder: str) -> None:
    """
    This will sample n amount of samples from the data points in ful_data
    """
    # list all file endings.
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  full_data_folder)

    print("getting data points")
    all_data_points_folder_path_names = get_all_data_path_names(
        full_data_folder)

    print("sampling data points")
    amount_sample_per_data_folder = (amount_samples //
                                     len(all_data_points_folder_path_names))
    selected_data: list[list[str]] = [
        random.sample(main_folder, amount_sample_per_data_folder) for
        main_folder in all_data_points_folder_path_names]

    print("flattening data points")
    flatten_selected_data: list[str] = []

    for foldered_data in selected_data:
        flatten_selected_data += foldered_data

    print("running normality tests")
    # test normality.
    if virtual_memory().total > 1600000000:
        threads: list[threading.Thread] = []
        for folder in selected_data:
            threads.append(threading.Thread(target=threaded_make_data_array,
                                            args=(folder, "sub folder",)))

        threads.append(threading.Thread(target=threaded_make_data_array,
                                        args=(flatten_selected_data,
                                              "whole data",)))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        for thread_output in multithread_data_test_output:
            name, data = thread_output

            normality = normaltest(data)

            plt.hist(data)
            plt.xlabel("Depth")
            plt.ylabel("Frequency")
            plt.title(f"{name} | norm: stat:{normality.statistic:0.2f}, " +
                      f"p:{normality.pvalue:0.2f}")
            plt.show()
    else:
        for folder in selected_data:
            test_dataset_normality(folder, "sub folder")

        test_dataset_normality(flatten_selected_data, "whole selected data")

    print("copying over data to subset data")
    output_data_directory = data_directory + "_subset"
    try:
        os.mkdir(output_data_directory)
        print(f"made data folder {full_data_folder + '_subset'}")
    except FileExistsError:
        rmtree(output_data_directory)
        os.mkdir(output_data_directory)

    # copy over the subset data in their own folders.
    for index, data_path in enumerate(flatten_selected_data):
        data_point_folder = os.path.join(output_data_directory, str(index))
        try:
            os.mkdir(data_point_folder)
            print(f"made data point {index}")
        except FileExistsError:
            rmtree(data_point_folder)
            os.mkdir(data_point_folder)

        for endings in [".png", "_depth.npy", "_depth_mask.npy"]:
            shutil.copy(data_path + endings, data_point_folder)


def main() -> None:
    """
    This main to to run the data subset maker on it's own
    """
    subset_full_dataset(10, "train")


if __name__ == '__main__':
    main()
