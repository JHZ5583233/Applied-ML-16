import numpy as np
import os
from PIL import Image
from typing import Literal


class DataLoader:
    def __init__(self, folder: Literal["train", "Val"]):
        """
        initialise the data loader to get data from subsettes train or val
        dataset.
        """
        self.folder = folder + "_subset"
        file_directory = __file__
        data_directory = os.path.join(os.path.split(file_directory)[0],
                                      self.folder)
