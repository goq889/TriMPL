from .base_dataset import BaseDataset


class UCF101(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "ucf101"
