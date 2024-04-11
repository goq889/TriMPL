from .base_dataset import BaseDataset


class DTD(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "dtd"
