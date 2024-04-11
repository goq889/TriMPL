import os
from .base_dataset import BaseDataset
from scipy.io import loadmat


class StanfordCars(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "stanford-cars"

    def read_data(self, cfg):
        root = os.path.abspath(cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.trn_val_dir = os.path.join(self.dataset_dir, "cars_train")
        self.test_dir = os.path.join(self.images_dir, "cars_test")

        split_file = f"split_{self.dataset_name}_trn{cfg.DATASET.TRN}_val{cfg.DATASET.VAL}.json"
        split_path = os.path.join(self.dataset_dir, split_file)

        if os.path.exists(split_path):
            train, val, test, categories = self._read_split(split_path, self.images_dir)
        else:
            trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
            test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
            meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")

            trainval, categories = self._read_by_anno_meta(self.trn_val_dir, trainval_file, meta_file)
            train, val = self._split_trainval(trainval)
            test = self._read_by_anno_meta(self.test_dir, test_file, meta_file)
            self._save_split(train, val, test, categories, split_path, self.images_dir)

        train = self._fewshot(cfg, train)
        return train, val, test, categories

    def _read_by_anno_meta(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        data = dict()
        categories = dict()

        for i in range(len(anno_file)):
            im_name = anno_file[i]["fname"][0]
            im_path = os.path.join(image_dir, im_name)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            v = data.setdefault(label, [])
            v.append({"im_path": im_path, "label": label, "category": classname})
            categories[label] = classname

        labels = sorted(data.keys())
        out = [data[label] for label in labels]
        categories = [categories[label] for label in labels]
        return out, categories
