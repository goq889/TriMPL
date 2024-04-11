import json
import os

import scipy.io as scio

from .base_dataset import BaseDataset


class OxfordFlowers(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "oxford-flowers"

    def read_data(self, cfg):
        root = os.path.abspath(cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, "images")

        label_path = os.path.join(self.dataset_dir, "imagelabels.mat")
        lab2cname_path = os.path.join(self.dataset_dir, "cat_to_name.json")

        split_file = f"split_{self.dataset_name}_trn{cfg.DATASET.TRN}_val{cfg.DATASET.VAL}.json"
        split_path = os.path.join(self.dataset_dir, split_file)

        if os.path.exists(split_path):
            train, val, test, categories = self._read_split(split_path, self.images_dir)
        else:
            dataset, categories = self._read_by_annotation(self.images_dir, label_path, lab2cname_path)
            train, val, test = self._split_data(dataset)
            self._save_split(train, val, test, categories, split_path, self.images_dir)

        train = self._fewshot(cfg, train)

        return train, val, test, categories

    @staticmethod
    def _read_by_annotation(images_dir, label_path, lab2cname_path):
        labels = scio.loadmat(label_path)["labels"][0]
        with open(lab2cname_path, mode="r", encoding="utf-8") as f:
            lab2cname = json.load(f)

        data = [[] for _ in range(len(lab2cname))]
        for idx, label in enumerate(labels):
            im_name = f"image_{str(idx + 1).zfill(5)}.jpg"
            im_path = os.path.join(images_dir, im_name)
            category = lab2cname[str(label)]
            label -= 1
            data[label].append({"im_path": im_path, "label": int(label), "category": category})

        labels = list(lab2cname.keys())
        labels.sort(key=lambda x: int(x))
        categories = [lab2cname[label] for label in labels]
        return data, categories
