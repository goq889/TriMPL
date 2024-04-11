import copy
import math

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip

from .base_dataset import TriMPLDataset
from .caltech_101 import Caltech101
from .dtd import DTD
from .eurosat import Eurosat
from .fgvc_aircraft import FGVCAircraft
from .food_101 import Food101
from .imagenet import ImageNet
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_s import ImageNetS
from .imagenet_v2 import ImageNetV2
from .oxford_flowers import OxfordFlowers
from .oxford_pets import OxfordPets
from .stanford_cars import StanfordCars
from .sun397 import SUN397
from .ucf101 import UCF101

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


_readers = {
    "caltech-101": Caltech101(),
    "dtd": DTD(),
    "eurosat": Eurosat(),
    "fgvc-aircraft": FGVCAircraft(),
    "food-101": Food101(),
    "imagenet": ImageNet(),
    "imagenet-a": ImageNetA(),
    "imagenet-r": ImageNetR(),
    "imagenet-s": ImageNetS(),
    "imagenet-v2": ImageNetV2(),
    "oxford-flowers": OxfordFlowers(),
    "oxford-pets": OxfordPets(),
    "stanford-cars": StanfordCars(),
    "sun397": SUN397(),
    "ucf101": UCF101(),
}


def _convert2rgb(image):
    return image.convert("RGB")


_eval_transforms = Compose([
    Resize(224, interpolation=BICUBIC),
    CenterCrop(224),
    _convert2rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

_train_transforms = Compose([
    RandomResizedCrop(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
    RandomHorizontalFlip(),
    _convert2rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def build_train_dataset(cfg):
    assert cfg.DATASET.TRAIN in _readers.keys()
    trn_reader = _readers[cfg.DATASET.TRAIN]
    train, val, _, categories = trn_reader.read_data(cfg)

    if cfg.TriMPL.MODE == "base_new":
        [trn_base, val_base], base_cate = _base_new_split(train, val, categories=categories, subsample="base")
        train, val, categories = trn_base, val_base, base_cate
    train = {"categories": categories, "dataset": TriMPLDataset([inst for instances in train for inst in instances], transforms=_train_transforms)}
    val = {"categories": categories, "dataset": TriMPLDataset([inst for instances in val for inst in instances], transforms=_eval_transforms)}
    return train, val


def build_test_dataset(cfg):
    assert cfg.TriMPL.MODE in ["normal", "base_new", "cross"]
    assert len(cfg.DATASET.TESTS) >= 1

    if cfg.TriMPL.MODE == "cross":
        tests = []
        for tst in cfg.DATASET.TESTS:
            assert tst in _readers.keys()
            tst_reader = _readers[tst]
            _, _, tst_dataset, categories = tst_reader.read_data(cfg)
            tests.append({
                "categories": categories,
                "dataset": TriMPLDataset([inst for instances in tst_dataset for inst in instances], transforms=_eval_transforms)
            })
        return tests

    tst = cfg.DATASET.TESTS[0]
    assert tst in _readers.keys()
    tst_reader = _readers[tst]
    _, _, tst_dataset, categories = tst_reader.read_data(cfg)
    if cfg.TriMPL.MODE == "normal":
        return [{
            "categories": categories,
            "dataset": TriMPLDataset([inst for instances in tst_dataset for inst in instances], transforms=_eval_transforms)
        }]

    [tst_base], base_cate = _base_new_split(tst_dataset, categories=categories, subsample="base")
    [tst_new], new_cate = _base_new_split(tst_dataset, categories=categories, subsample="new")
    return [
        {"categories": base_cate, "dataset": TriMPLDataset([inst for instances in tst_base for inst in instances], transforms=_eval_transforms)},
        {"categories": new_cate, "dataset": TriMPLDataset([inst for instances in tst_new for inst in instances], transforms=_eval_transforms)}
    ]


def _base_new_split(*datasets, categories, subsample):
    assert subsample in ["all", "base", "new"]
    if subsample == "all":
        return datasets

    dataset = datasets[0]
    n_cls = len(dataset)
    b_cls = math.ceil(n_cls / 2)

    out = []
    for dataset in datasets:
        if subsample == "base":
            out.append(copy.deepcopy(dataset[:b_cls]))
        else:
            new_dataset = copy.deepcopy(dataset[b_cls:])
            for new_label, instances in enumerate(new_dataset):
                for instance in instances:
                    instance["label"] = new_label
            out.append(new_dataset)

    cate = categories[:b_cls] if subsample == "base" else categories[b_cls:]
    return out, cate
