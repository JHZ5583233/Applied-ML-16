from pathlib import Path
from typing import Tuple
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
import numpy as np


from data.data_loader import DataLoader as OriginalDataLoader


class ZoeDepthDataset(Dataset):
    """
    PyTorch Dataset wrapper around the original custom DataLoader
    (from data.data_loader import DataLoader).
    """
    def __init__(self, split: str):
        if split.lower() not in {"train", "val"}:
            raise ValueError("Split must be 'train' or 'val'")
        self.loader = OriginalDataLoader(split.lower())
        self.total_samples = len(self.loader.data_paths)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        self.loader.data_index = idx
        data = self.loader.get_data()

        image_np = data[0]
        depth_np = data[1]

        image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth_np).float()

        return image, depth
