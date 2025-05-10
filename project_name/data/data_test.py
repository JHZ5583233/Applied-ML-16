import numpy as np
import matplotlib.pyplot as plt

file_end = "_depth.npy"


def test_dataset_normality(data: list[str]) -> None:
    whole_data = np.array([])
    max_n = 0

    for data_point in data:
        matrix: np.ndarray = np.load(data_point + file_end)
        max_n = max(max_n, matrix.max())
        whole_data = np.concatenate([whole_data, matrix.flatten()])

    plt.hist(whole_data)
    plt.show()


def main():
    test_dataset_normality()


if __name__ == '__main__':
    main()
