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
import torch.optim as optim
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
from data_loader import CustomDataset
from op_utils import train_loop, val_loop, inference
from efficientnet_pytorch import  EfficientNet


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

        self.fc = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(),
                                nn.Linear(512, 1),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.resnetmodel(x)
        return self.fc(x)



if __name__ == '__main__':
    csv_file = r'D:\jaehwan\98.dacon\mammography\csv\test.csv'
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

    patients_ids = df.patient_id.unique()
    val_df = df.loc[df.patient_id.isin(patients_ids), :].copy()
    val_transforms = transforms.Compose([transforms.ToTensor()])
    val_dataset = CustomDataset(val_df, transform=val_transforms, mode='inference')


    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        pin_memory=True)

    model = torchvision.models.resnet18(pretrained=True)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # trained_model = r'D:\jaehwan\98.dacon\mammography\src\workspace\model\checkpoints\best_effcientnet_3_epoch_train_loss_0.656_train_acc_0.637_val_loss_0.654_val_acc_0.675.pth'
    trained_model = r'D:\jaehwan\98.dacon\mammography\src\workspace\model\checkpoints\best_effcientnet_15_epoch_train_loss_0.564_train_acc_0.848_val_loss_0.640_val_acc_0.692.pth'
    trained_model =r'D:\jaehwan\98.dacon\mammography\src\workspace\model\checkpoints\best_effcientnet_26_epoch_train_loss_0.523_train_acc_0.945_val_loss_0.661_val_acc_0.700.pth'
    model.load_state_dict(torch.load(trained_model), strict=False)

    ## TODO: INFERENCE

    preds = inference(val_loader=val_dataloader, model=model, device=device)

    ## TODO: SUBMISSON

    submit = pd.read_csv(r'D:\jaehwan\98.dacon\mammography\csv\submit.csv')
    submit['N_category'] = preds
    submit.to_csv(r'D:\jaehwan\98.dacon\mammography\src\workspace\result\submit.csv', index=False)

    print('complete!')