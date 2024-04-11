from .base_dataset import BaseDataset


class Eurosat(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "eurosat"
        self.cnames = {
            "AnnualCrop": "Annual Crop Land",
            "Forest": "Forest",
            "HerbaceousVegetation": "Herbaceous Vegetation Land",
            "Highway": "Highway or Road",
            "Industrial": "Industrial Buildings",
            "Pasture": "Pasture Land",
            "PermanentCrop": "Permanent Crop Land",
            "Residential": "Residential Buildings",
            "River": "River",
            "SeaLake": "Sea or Lake",
        }
