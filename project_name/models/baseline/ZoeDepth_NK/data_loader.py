from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "Data"


class ZoeDepthDataset(Dataset):
    """
    Dataset for ZoeDepth model:
    Loads RGB images and corresponding depth ground truths.
    """
    def __init__(self, split: str):
        self.folder = DATA_DIR / split
        self.samples = sorted(d for d in self.folder.iterdir() if d.is_dir())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.samples[idx]
        image_path = next(sample_dir.glob("*.png"))
        depth_path = next(sample_dir.glob("*_depth.npy"))

        image = Image.open(image_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        depth_gt = np.load(depth_path).astype(np.float32)
        depth_gt = torch.from_numpy(depth_gt)

        return image, depth_gt
