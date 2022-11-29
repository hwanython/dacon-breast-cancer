import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.optim
from torchvision import datasets, models, transforms
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from data_loader import CustomDataset
from torch.utils.data import DataLoader
from op_utils import train_loop, val_loop

def get_values(value):
    return value.values.reshape(-1, 1)

def f1_score(preds, targets):
    tp = (preds * targets).sum().to(torch.float32)
    fp = ((1 - targets) * preds).sum().to(torch.float32)
    fn = (targets * (1 - preds)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    return f1_score


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        # self.efficientnet = torchvision.models.efficientnet_v2_m(pretrained=True)
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.embedding = nn.Linear(1000, 512)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x


class TabularFeatureExtractor(nn.Module):
    def __init__(self):
        super(TabularFeatureExtractor, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_features=23, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512)
        )

    def forward(self, x):
        x = self.embedding(x)
        return x


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.img_feature_extractor = CNNmodel()
        self.tabular_feature_extractor = TabularFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid(),
        )
        # self.classifier = nn.Sequential(nn.Dropout(0.5),
        #                         nn.Linear(1024, 1024), nn.ReLU(),
        #                         nn.Dropout(0.2),
        #                         nn.Linear(1024, 512),
        #                         nn.Dropout(0.1),
        #                         nn.Linear(512, 1),
        #                         nn.Sigmoid())

    def forward(self, img, tabular):
        img_feature = self.img_feature_extractor(img)
        tabular_feature = self.tabular_feature_extractor(tabular)
        feature = torch.cat([img_feature, tabular_feature], dim=-1)
        output = self.classifier(feature)
        return output


if __name__ == '__main__':
    csv_file = r'D:\jaehwan\98.dacon\mammography\csv\train.csv'
    df = pd.read_csv(csv_file, encoding='cp949')
    df['암의 장경'] = df['암의 장경'].fillna(df['암의 장경'].mean())
    df = df.fillna(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 8
    OUTPUT_PATH = ""
    MODEL_PATH = "../model/"
    LOSSES_PATH = "../model/"
    torch.manual_seed(0)
    np.random.seed(0)

    patients = df.ID.unique()
    train_ids, val_ids = train_test_split(patients,
                                          test_size=0.4,
                                          random_state=0)

    train_df = df.loc[df.ID.isin(train_ids), :].copy()
    val_df = df.loc[df.ID.isin(val_ids), :].copy()

    numeric_cols = ['나이', '암의 장경', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2_SISH_ratio']
    ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']

    for col in train_df.columns:
        if col in ignore_cols:
            continue
        if col in numeric_cols:
            scaler = StandardScaler()
            train_df[col] = scaler.fit_transform(get_values(train_df[col]))
            val_df[col] = scaler.transform(get_values(val_df[col]))
        else:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(get_values(train_df[col]))
            val_df[col] = le.transform(get_values(val_df[col]))

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor()])

    val_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = CustomDataset(train_df, transform=train_transforms, mode='train')
    val_dataset = CustomDataset(val_df, transform=val_transforms, mode='train')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True,
        pin_memory=True)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True)

    model = nn.DataParallel(ClassificationModel())
    # model = torchvision.models.inception_v3(pretrained=True)
    model = model.to(device)
    # model.eval()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss().to(device)
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
