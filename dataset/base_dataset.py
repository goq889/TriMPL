import json
import os.path
import os.path
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class BaseDataset:
    def __init__(self):
        self.dataset_dir = ""
        self.images_dir = ""
        self.dataset_name = "base-dataset"
        self.ignored = []
        self.cnames = dict()

    def read_data(self, cfg):
        root = os.path.abspath(cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, "images")

        split_file = f"split_{self.dataset_name}_trn{cfg.DATASET.TRN}_val{cfg.DATASET.VAL}.json"
        split_path = os.path.join(self.dataset_dir, split_file)
        if os.path.exists(split_path):
            train, val, test, categories = self._read_split(split_path, self.images_dir)
        else:
            dataset, categories = self._read_by_category_dir(self.images_dir)
            train, val, test = self._split_data(dataset)
            self._save_split(train, val, test, categories, split_path, self.images_dir)

        train = self._fewshot(cfg, train)

        return train, val, test, categories

    def _fewshot(self, cfg, train):
        if cfg.DATASET.NUM_SHOTS <= 0:
            return train

        fewshot_file = f"split_{self.dataset_name}_trn{cfg.DATASET.TRN}_val{cfg.DATASET.VAL}_shots{cfg.DATASET.NUM_SHOTS}_seed{cfg.SEED}.json"
        fewshot_path = os.path.join(self.dataset_dir, fewshot_file)
        if os.path.exists(fewshot_path):
            train, _, _, _ = self._read_split(fewshot_path, self.images_dir)
        else:
            train = self._generate_fewshot(train, cfg.DATASET.NUM_SHOTS)
            self._save_split(train, [], [], [], fewshot_path, self.images_dir)
        return train

    def _read_by_category_dir(self, images_dir):
        data = []
        categories = filter(lambda name: os.path.isdir(os.path.join(images_dir, name)), os.listdir(images_dir))
        categories = list(filter(lambda c: c not in self.ignored, categories))
        categories.sort()

        ret_categories = []
        for label, category in enumerate(categories):
            category_dir = os.path.join(images_dir, category)
            images = filter(lambda name: os.path.isfile(os.path.join(category_dir, name)), os.listdir(category_dir))
            category = self.cnames[category] if category in self.cnames else category
            category = " ".join(category.split("_")).lower()
            category_data = [{"im_path": os.path.join(category_dir, im), "label": label, "category": category} for im in
                             images]
            ret_categories.append(category)
            data.append(category_data)
        return data, ret_categories

    @staticmethod
    def _read_split(split_path, prefix):
        with open(split_path, mode="r", encoding="utf-8") as split_json:
            split = json.load(split_json)

        def _convert(dataset):
            out = []
            for instances in dataset:
                category_out = []
                for im_path, label, category in instances:
                    category_out.append(
                        {"im_path": os.path.join(prefix, im_path), "label": label, "category": category})
                out.append(category_out)
            return out

        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        categories = split["categories"]
        return train, val, test, categories

    @staticmethod
    def _split_data(dataset, p_trn=0.5, p_val=0.2):
        p_tst = 1 - p_trn - p_val
        train, val, test = [], [], []
        for instances in dataset:
            random.shuffle(instances)
            n_total = len(instances)
            n_trn = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_tst = n_total - n_trn - n_val
            assert n_trn > 0 and n_val > 0 and n_tst > 0

            train.append(instances[:n_trn])
            val.append(instances[n_trn:n_trn + n_val])
            test.append(instances[n_trn + n_val:])

        return train, val, test

    @staticmethod
    def _split_trainval(dataset, p_val=0.2):
        p_trn = 1 - p_val
        train, val = [], []
        for instances in dataset:
            random.shuffle(instances)
            n_total = len(instances)
            n_trn = round(n_total * p_trn)
            n_val = n_total - n_trn
            assert n_trn > 0 and n_val > 0

            train.append(instances[:n_trn])
            val.append(instances[n_trn:])

        return train, val

    @staticmethod
    def _save_split(train, val, test, categories, split_path, prefix):
        def _convert(dataset):
            out = []
            for instances in dataset:
                category_out = []
                for instance in instances:
                    im_path = instance["im_path"].replace(prefix, "")
                    im_path = im_path[1:] if im_path.startswith('/') else im_path
                    label = instance["label"]
                    category = instance["category"]
                    category_out.append([im_path, label, category])
                out.append(category_out)
            return out

        train_sv = _convert(train)
        val_sv = _convert(val)
        test_sv = _convert(test)
        split = {"train": train_sv, "val": val_sv, "test": test_sv, "categories": categories}

        with open(split_path, mode="w", encoding="utf-8") as f:
            json.dump(split, f)

    @staticmethod
    def _generate_fewshot(dataset, n_shots):
        fs_dataset = []

        for instances in dataset:
            if len(instances) >= n_shots:
                sampled_instances = random.sample(instances, n_shots)
            else:
                sampled_instances = random.choices(instances, k=n_shots)

            fs_dataset.append(sampled_instances)
        return fs_dataset


class TriMPLDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        assert transforms is not None
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        instance = self.dataset[idx]
        im_path = instance["im_path"]
        label = instance["label"]
        category = instance["category"]
        image = Image.open(im_path)
        image = self.transforms(image)
        label = torch.tensor(label)
        return image, label, category
