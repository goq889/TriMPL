import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
import trimpl
import trainer

from torch.utils.data import DataLoader
from configs import config
from dataset.build import build_train_dataset, build_test_dataset


def main(args):
    cfg = config.get_cfg_defaults()

    if args.config_file:
        cfg.merge_from_file(os.path.join("configs", args.config_file))

    if args.data:
        cfg.DATASET.ROOT = args.data

    if args.train:
        cfg.DATASET.TRAIN = args.train

    if args.tests:
        cfg.DATASET.TESTS = args.tests

    if args.mode:
        cfg.TriMPL.MODE = args.mode

    if args.shot > 0:
        cfg.DATASET.NUM_SHOTS = args.shot

    if args.seed:
        cfg.SEED = args.seed

    if args.warm:
        cfg.OPTIM.WARMUP_TYPE = args.warm

    if args.output:
        cfg.OUTPUT_DIR = args.output

    if args.overall:
        cfg.TriMPL.OVERALL = args.overall

    _setup_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.eval_only:
        tests = build_test_dataset(cfg)
        model_path = args.model_path
        for tst in tests:
            model = trimpl.load_model(model_path, cfg, tst["categories"])
            tst_dataloader = DataLoader(
                dataset=tst["dataset"],
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.DATALOADER.NUM_WORKERS
            )
            acc = trainer.trimpl_test(model, tst_dataloader, device)
            print(acc)
    else:
        if cfg.TriMPL.MODE in ["normal", "base_new"]:
            cfg.DATASET.TESTS = [cfg.DATASET.TRAIN]

        train, val = build_train_dataset(cfg)
        model = trimpl.build_model(cfg, train["categories"], device)
        trn_dataloader = DataLoader(
            dataset=train["dataset"],
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS
        )
        val_dataloader = DataLoader(
            dataset=val["dataset"],
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS
        )
        trainer.trimpl_train(cfg, model, trn_dataloader, val_dataloader, device)


def _setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="", help="path to data")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--train", type=str, default="", help="train dataset")
    parser.add_argument("--mode", type=str, default="", help="normal, base_new, cross")
    parser.add_argument("--shot", type=int, default=0, help="n-shots")
    parser.add_argument("--seed", type=int, default=2023, help="only positive value enables a fixed seed")
    parser.add_argument("--warm", type=str, default="", help="warm up type")
    parser.add_argument("--output", type=str, default="", help="model saved dir")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-path", type=str, default="", help="load model weights")
    parser.add_argument("--tests", action="append", help="test dataset")
    args = parser.parse_args()
    main(args)
