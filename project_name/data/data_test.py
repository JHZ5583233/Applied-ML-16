import numpy as np
import matplotlib.pyplot as plt

file_end = "_depth.npy"


def test_dataset_normality(data: list[str]) -> None:
    """
    plots the histogram of given depth maps.

    data: list of path names to the data point name
    """
    whole_data = np.array([])
    max_n = 0

    for data_point in data:
        matrix: np.ndarray = np.load(data_point + file_end)
        max_n = max(max_n, matrix.max())
        whole_data = np.concatenate([whole_data, matrix.flatten()])

    plt.hist(whole_data)
    plt.show()
