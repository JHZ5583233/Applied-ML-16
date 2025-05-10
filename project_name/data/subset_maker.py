import os
import random
import shutil


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
        datapoint_directories = [main_folder]
        for _ in range(3):
            new_datapoint_directories = []

            for path in datapoint_directories:
                sub_folders = os.listdir(path)

                for folder in sub_folders:
                    # TODO get rid of file extension and type description.
                    new_datapoint_directories.append(os.path.join(path,
                                                                  folder))

            # get rid of all duplicates
            datapoint_directories = list(set(new_datapoint_directories))

        all_datapoint_folder_pathnames.append(datapoint_directories)

    return all_datapoint_folder_pathnames


def get_train_data_folders() -> list[str]:
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  "subset_data")

    return os.listdir(data_directory)


def subset_full_dataset(amount_samples: int):
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

    # TODO list all file endings.
    file_endings = []
    # copy over the subsetted data in their own folders.
    for index, data in enumerate(flatten_selected_data):
        data_point_folder = os.path.join(data_directory, str(index))
        try:
            os.mkdir(data_point_folder)
            print(f"made data point {index}")
        except FileExistsError:
            os.rmdir(data_point_folder)
            os.mkdir(data_point_folder)

        for endings in file_endings:
            shutil.copy(os.path.join(data, endings), data_point_folder)


def main():
    pass


if __name__ == '__main__':
    main()
