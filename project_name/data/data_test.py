import numpy as np

from subset_maker import get_all_data_pathnames


def test_dataset_normality():
    in_out_door_data = get_all_data_pathnames()

    for data in in_out_door_data:
        for point in data:
            print(np.load(point))


def main():
    test_dataset_normality()


if __name__ == '__main__':
    main()
