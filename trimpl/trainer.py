import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import _LRScheduler


class LinearCosineAnnealing(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epoch, st_lr=1e-5, last_epoch=-1, verbose=False):
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max - warmup_epoch, 0, last_epoch,
                                                                 verbose)
        self.warmup_epoch = warmup_epoch
        self.st_lr = st_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.cosine.get_last_lr()
        if self.last_epoch == 0:
            return [self.st_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.cosine.step(epoch)
            self._last_lr = self.cosine.get_last_lr()
        else:
            super().step(epoch)


class ConstantCosineAnnealing(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_epoch, const_lr=1e-5, last_epoch=-1, verbose=False):
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max - warmup_epoch, 0, last_epoch,
                                                                 verbose)
        self.warmup_epoch = warmup_epoch
        self.const_lr = const_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.cosine.get_last_lr()
        else:
            return [self.const_lr for _ in self.base_lrs]

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.cosine.step(epoch)
            self._last_lr = self.cosine.get_last_lr()
        else:
            super().step(epoch)


class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        bs = features.shape[0]
        features = features / features.norm(dim=-1, keepdim=True)
        features_t = features.transpose(1, 2)
        cos_abs = torch.abs(features @ features_t)
        cos_abs = torch.triu(cos_abs, diagonal=1)
        cos_abs_sum = torch.sum(cos_abs)
        loss_orth = cos_abs_sum / bs
        return loss_orth


def trimpl_train(cfg, model, trn_loader, val_loader, device):
    lr = cfg.OPTIM.LR
    weight_decay = cfg.OPTIM.WEIGHT_DECAY
    momentum = cfg.OPTIM.MOMENTUM
    sgd_dampening = cfg.OPTIM.SGD_DAMPENING
    sgd_nesterov = cfg.OPTIM.SGD_NESTEROV

    if isinstance(model, nn.DataParallel):
        parameters = [
            model.module.custom_text_encoder.general_prompt,
            model.module.custom_text_encoder.feature_descriptors
        ]
        mix_parameters = [model.module.custom_text_encoder.mix_alpha, model.module.custom_text_encoder.mix_beta]
    else:
        parameters = [
            model.custom_text_encoder.general_prompt,
            model.custom_text_encoder.feature_descriptors
        ]
        mix_parameters = [model.module.custom_text_encoder.mix_beta]

    optimizer = torch.optim.SGD(
        [{"params": parameters}, {"params": mix_parameters, "lr": 0.1}],
        lr=lr,
        momentum=momentum,
        dampening=sgd_dampening,
        weight_decay=weight_decay,
        nesterov=sgd_nesterov,
    )

    max_epoch = cfg.OPTIM.MAX_EPOCH
    warm_epoch = cfg.OPTIM.WARMUP_EPOCH
    warm_min_lr = cfg.OPTIM.MIN_LR

    if cfg.OPTIM.WARMUP_TYPE == "LINEAR":
        scheduler = LinearCosineAnnealing(optimizer, max_epoch, warm_epoch, warm_min_lr)
    else:
        scheduler = ConstantCosineAnnealing(optimizer, max_epoch, warm_epoch, warm_min_lr)

    CLS_LOSS = CrossEntropyLoss()
    ORTH_LOSS = OrthogonalLoss()
    orth_lambda = cfg.TriMPL.ORTH_LAMBDA

    best_acc = 0
    for epoch in range(cfg.OPTIM.MAX_EPOCH):
        total = 0
        right = 0
        loss_cls_record = []
        model.train()
        for images, labels, categories in trn_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, descriptors = model(images)
            loss_cls = CLS_LOSS(logits, labels)
            loss_cls_record.append(loss_cls.detach().item())
            loss = loss_cls
            if orth_lambda > 0:
                loss_orth = orth_lambda * ORTH_LOSS(descriptors)
                loss = loss + loss_orth
            loss.backward()
            optimizer.step()
            total += images.shape[0]
            right += (torch.argmax(logits, dim=1) == labels).sum().item()
            print("acc " + f"{right / total:.5f}")
            print("loss " + f"{sum(loss_cls_record) / len(loss_cls_record):.5f}")

        loss_v, acc_v = _valid(model, val_loader, device)
        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        model_path = (f"epoch{epoch+1}|"
                      + f"mode_{cfg.TriMPL.MODE}|"
                      + f"orthL_{cfg.TriMPL.ORTH_LAMBDA}|"
                      + f"seed_{cfg.SEED}|"
                      + f"acc_v{acc_v}.pt")
        model_path = os.path.join(os.path.abspath(cfg.OUTPUT_DIR), model_path)
        torch.save(state_dict, model_path)
        scheduler.step()


def _valid(model, val_loader, device):
    model.eval()
    cls_loss = CrossEntropyLoss()
    loss_record = []
    total = 0
    right = 0
    for images, labels, categories in val_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits, _ = model(images)
            loss = cls_loss(logits, labels)
            loss_record.append(loss.detach().item())
            total += images.shape[0]
            right += (torch.argmax(logits, dim=1) == labels).sum().item()

    return sum(loss_record) / len(loss_record), right / total

def trimpl_test(model, tst_loader, device):
    model = model.to(device)
    model.eval()
    total = 0
    right = 0
    for images, labels, categories in tst_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            logits, _ = model(images)
            total += images.shape[0]
            right += (torch.argmax(logits, dim=1) == labels).sum().item()

    return right / total
