from typing import Tuple, Any
import numpy as np
from data.data_loader import DataLoader

class LinearRegressionDataset:

    
    """Dataset loader for Linear Regression training & testing data."""

    def __init__(self, split: str):
        """
        Args:
            split (str): "train" or "test"
        """
        mapped_split = "Val" if split.lower() == "test" else "train"
        self.data_loader = DataLoader(mapped_split)

        self.X = []
        self.y = []

        for _ in range(len(self.data_loader.data_paths)):
            image, depth = self.data_loader.get_data()
            self.X.append(image.flatten())
            self.y.append(depth.mean())

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self) -> int:
        """retur the lenght of the data set"""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return an item at a specidied index"""
        return self.X[idx], self.y[idx]

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return full dataset as (X, y) numpy arrays."""
        return self.X, self.y
