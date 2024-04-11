import os

from .base_dataset import BaseDataset


class SUN397(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "sun397"

    def read_data(self, cfg):
        root = os.path.abspath(cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, "images")
        anno_dir = os.path.join(self.dataset_dir, "Partitions")

        split_file = f"split_{self.dataset_name}_trn{cfg.DATASET.TRN}_val{cfg.DATASET.VAL}.json"
        split_path = os.path.join(self.dataset_dir, split_file)

        if os.path.exists(split_path):
            train, val, test, categories = self._read_split(split_path, self.images_dir)
        else:
            categories = []
            with open(os.path.join(anno_dir, "ClassName.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()[1:]
                    categories.append(line)
            categories.sort()

            trainval_txt = os.path.join(anno_dir, "Training_01.txt")
            trainval = self.read_by_anno(categories, trainval_txt, self.images_dir)
            test_txt = os.path.join(anno_dir, "Testing_01.txt")
            test = self.read_by_anno(categories, test_txt, self.images_dir)
            train, val = self._split_trainval(trainval)

            categories = [" ".join([" ".join(name.split("_")) for name in c.split("/")[1:]]) for c in categories]

            self._save_split(train, val, test, categories, split_path, self.images_dir)

        train = self._fewshot(cfg, train)
        return train, val, test, categories

    @staticmethod
    def read_by_anno(categories, anno_txt, images_dir):
        data = [[] for _ in range(len(categories))]
        with open(anno_txt, mode="r", encoding="utf-8") as f:
            for line in f:
                im_name = line.strip()[1:]
                im_path = os.path.join(images_dir, im_name)

                category = os.path.dirname(im_name)
                label = categories.index(category)

                names = category.split("/")[1:]
                names = [" ".join(name.split("_")) for name in names]
                category = " ".join(names)

                data[label].append({"im_path": im_path, "label": label, "category": category})
        return data
