from pathlib import Path
from typing import Tuple, List, Any
import numpy as np

from data.path_grapper import get_train_data_folders  # Assuming this is your data loader helper


class LinearRegressionDataset:
    """Dataset loader for Linear Regression training & testing data."""
    
    def __init__(self, split: str):
        """
        Args:
            split (str): "train" or "test"
        """
        # Use your existing helper to load folder data
        data = get_train_data_folders(split)
        self.X = data[0]
        self.y = data[1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        return self.X[idx], self.y[idx]

    def get_all(self) -> Tuple[List[Any], List[Any]]:
        """Return full dataset as (X, y) lists."""
        return self.X, self.y
