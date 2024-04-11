import os

from .base_dataset import BaseDataset


class FGVCAircraft(BaseDataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "fgvc-aircraft"

    def read_data(self, cfg):
        root = os.path.abspath(cfg.DATASET.ROOT)
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.images_dir = os.path.join(self.dataset_dir, "images")
        variants_path = os.path.join(self.dataset_dir, "variants.txt")

        categories = []
        with open(variants_path, mode="r", encoding="utf-8") as f:
            for line in f:
                categories.append(line.strip())
        categories.sort()

        trn_txt_path = os.path.join(self.dataset_dir, "images_variant_train.txt")
        train = FGVCAircraft.read_by_txt(self.images_dir, trn_txt_path, categories)
        val_txt_path = os.path.join(self.dataset_dir, "images_variant_val.txt")
        val = FGVCAircraft.read_by_txt(self.images_dir, val_txt_path, categories)
        tst_txt_path = os.path.join(self.dataset_dir, "images_variant_test.txt")
        test = FGVCAircraft.read_by_txt(self.images_dir, tst_txt_path, categories)

        train = self._fewshot(cfg, train)

        return train, val, test, categories

    @staticmethod
    def read_by_txt(images_dir, txt_path, categories):
        data = [[] for _ in range(len(categories))]
        with open(txt_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(" ")
                im_name = line[0] + ".jpg"
                im_path = os.path.join(images_dir, im_name)
                category = " ".join(line[1:])
                label = categories.index(category)
                assert label != -1
                data[label].append({"im_path": im_path, "label": label, "category": category})
        return data
