import matplotlib.pyplot as plt
from numpy import array, ndarray, load, concatenate
from scipy.stats import normaltest


def test_dataset_normality(data: list[str], name: str) -> None:
    """
    plots the histogram of given depth maps.

    data: list of path names to the data point name
    """
    whole_data = array([])
    max_n = 0

    for data_point in data:
        matrix: ndarray = load(data_point + "_depth.npy")
        max_n = max(max_n, matrix.max())
        whole_data = concatenate([whole_data, matrix.flatten()])

    normality = normaltest(whole_data)

    plt.hist(whole_data)
    plt.xlabel("Depth")
    plt.ylabel("Frequency")
    plt.title(f"{name} | norm: stat:{normality.statistic:0.2f}, " +
              f"p:{normality.pvalue:0.2f}")
    plt.show()
