import copy
import os

from imagenet import ImageNet
from .base_dataset import BaseDataset


class ImageNetV2(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "imagenet-v2"
        self.cnames = {}

    def read_data(self, cfg):
        root = os.path.abspath(cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, "images")

        classnames_path = os.path.join(self.dataset_dir, "classnames.txt")
        self.cnames = ImageNet.read_classnames(classnames_path)
        for i, (key, value) in enumerate(self.cnames.items()):
            new_key = str(i)  # 将索引转换为数字字符串
            self.cnames[new_key] = value

        test, categories = self._read_by_category_dir(self.images_dir)
        train = copy.deepcopy(test)
        val = copy.deepcopy(test)
        return train, val, test, categories
