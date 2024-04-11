import copy
import os
from collections import OrderedDict
from .base_dataset import BaseDataset


class ImageNet(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "imagenet"
        self.cnames = {}

    def read_data(self, cfg):
        root = os.path.abspath(cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, "images")

        classnames_path = os.path.join(self.dataset_dir, "classnames.txt")
        self.cnames = self.read_classnames(classnames_path)

        train_dir = os.path.join(self.images_dir, "train")
        train, categories_trn = self._read_by_category_dir(train_dir)

        val_dir = os.path.join(self.images_dir, "val")
        val, categories_val = self._read_by_category_dir(val_dir)

        assert categories_trn == categories_val

        train = self._fewshot(cfg, train)
        test = copy.deepcopy(val)
        return train, val, test, categories_trn

    @staticmethod
    def read_classnames(text_file):
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames
