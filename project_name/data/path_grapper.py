import os

file_endings = {"rgb": ".png",
                "depth": "_depth.npy",
                "depth_mask": "_depth_mask.npy"}


def get_all_data_pathnames() -> list[list[str]]:
    """
    get all path names of the data points name in the full data folder
    """
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
    """
    get all the folder names in the subset data folder
    """
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  "subset_data")

    return os.listdir(data_directory)
