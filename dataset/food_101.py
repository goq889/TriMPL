from .base_dataset import BaseDataset


class Food101(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "food-101"
