from .base_dataset import BaseDataset


class Caltech101(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "caltech-101"
        self.ignored = ["BACKGROUND_Google", "Faces_easy"]
        self.cnames = {
            "airplanes": "airplane",
            "Faces": "face",
            "Leopards": "leopard",
            "Motorbikes": "motorbike",
        }
