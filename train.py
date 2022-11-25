import os
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import timm
import numpy as np
import seaborn as sns

sns.set()
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from glob import glob
from skimage.io import imread
from os import listdir

import time
import copy
from tqdm import tqdm_notebook as tqdm
from efficientnet_pytorch import EfficientNet
from data_loader import CustomDataset
from op_utils import train_loop, val_loop


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def f1_score(preds, targets):
    tp = (preds * targets).sum().to(torch.float32)
    fp = ((1 - targets) * preds).sum().to(torch.float32)
    fn = (targets * (1 - preds)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    return f1_score


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.resnetmodel = EfficientNet.from_pretrained('efficientnet-b7')

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(1000, 1024), nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(1024, 512),
                                nn.Dropout(0.1),
                                nn.Linear(512, 1),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.resnetmodel(x)
        return self.fc(x)


if __name__ == '__main__':
    csv_file = r'D:\jaehwan\98.dacon\mammography\csv\train.csv'
    df = pd.read_csv(csv_file, encoding_errors='ignore')
    id = df.iloc[:, 0].values
    imgs_path = df.iloc[:, 1].values
    label = df.iloc[:, -1].values
    df = pd.DataFrame([x for x in zip(id.tolist(), imgs_path.tolist(), label.tolist())],
                      columns=["patient_id", "path", "labels"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 8
    OUTPUT_PATH = ""
    MODEL_PATH = "../model/"
    LOSSES_PATH = "../model/"
    torch.manual_seed(0)
    np.random.seed(0)

    patients = df.patient_id.unique()
    train_ids, val_ids = train_test_split(patients,
                                          test_size=0.4,
                                          random_state=0)

    train_df = df.loc[df.patient_id.isin(train_ids), :].copy()
    val_df = df.loc[df.patient_id.isin(val_ids), :].copy()

    train_transforms = transforms.Compose([
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomAutocontrast(),
        transforms.ToTensor()])

    val_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = CustomDataset(train_df, transform=train_transforms, mode='train')
    val_dataset = CustomDataset(val_df, transform=val_transforms, mode='train')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=True,
        pin_memory=True)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        pin_memory=True)

    # model = CustomModel()
    model = torchvision.models.resnet18(pretrained=True)
    # model = torchvision.models.inception_v3(pr)

    # model.classifier.fc = nn.Sequential(
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.Dropout(0.1),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    model = model.to(device)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # base_optimizer = torch.optim.SGD
    # optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    ## TODO: TRAINING
    EPOCHS = 1000
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, model = train_loop(train_loader=train_dataloader, model=model, criterion=criterion,
                                                  optimizer=optimizer, epoch=epoch, device=device)
        val_loss, val_acc, _ = val_loop(val_loader=val_dataloader, model=model, criterion=criterion, epoch=epoch,
                                        device=device)

        print(
            f'{epoch} Epoch | train loss:{train_loss:.3f} | train acc:{train_acc:.3f} | val loss:{val_loss:.3f} | val acc:{val_acc:.3f}')

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       './model/checkpoints/best_effcientnet_{0}_epoch_train_loss_{1:.3f}_train_acc_{2:.3f}_val_loss_{3:.3f}_val_acc_{4:.3f}.pth'.format(
                           epoch, train_loss, train_acc, val_loss, val_acc))
