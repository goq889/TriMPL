import os

from .base_dataset import BaseDataset


class OxfordPets(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "oxford-pets"

    def read_data(self, cfg):
        root = os.path.abspath(cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, "images")
        anno_dir = os.path.join(self.dataset_dir, "annotations")

        split_file = f"split_{self.dataset_name}_trn{cfg.DATASET.TRN}_val{cfg.DATASET.VAL}.json"
        split_path = os.path.join(self.dataset_dir, split_file)

        if os.path.exists(split_path):
            train, val, test, categories = self._read_split(split_path, self.images_dir)
        else:
            trainval_txt = os.path.join(anno_dir, "trainval.txt")
            trainval, categories = self.read_by_annotation(trainval_txt, self.images_dir)
            test_txt = os.path.join(anno_dir, "test.txt")
            test, _ = self.read_by_annotation(test_txt, self.images_dir)

            train, val = self._split_trainval(trainval)
            self._save_split(train, val, test, categories, split_path, self.images_dir)

        train = self._fewshot(cfg, train)
        return train, val, test, categories

    @staticmethod
    def read_by_annotation(anno_txt, images_dir):
        data = dict()
        categories = dict()
        with open(anno_txt, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(" ")
                im_name, label = line[0], line[1]
                category = " ".join(im_name.split("_")[:-1]).lower()
                im_name += ".jpg"
                im_path = os.path.join(images_dir, im_name)
                label = int(label) - 1
                v = data.setdefault(label, [])
                v.append({"im_path": im_path, "label": label, "category": category})
                categories[label] = category

        labels = sorted(data.keys())
        out = [data[label] for label in labels]
        categories = [categories[label] for label in labels]
        return out, categories
