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
from op_utils import inference
from torch.utils.data import DataLoader

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
    csv_file = r'D:\jaehwan\98.dacon\mammography\csv\test.csv'
    df = pd.read_csv(csv_file, encoding='cp949')

    # set nan or 0 data
    df['암의 장경'] = df['암의 장경'].fillna(df['암의 장경'].mean())
    df = df.fillna(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 8
    OUTPUT_PATH = ""
    MODEL_PATH = "../model/"
    LOSSES_PATH = "../model/"
    torch.manual_seed(0)
    np.random.seed(0)

    # patients_ids = df.patient_id.unique()
    patients_ids = df.ID.unique()
    val_df = df.loc[df.ID.isin(patients_ids), :].copy()
    val_transforms = transforms.Compose([transforms.ToTensor()])

    numeric_cols = ['나이', '암의 장경', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2_SISH_ratio']
    ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']

    for col in val_df.columns:
        if col in ignore_cols:
            continue
        if col in numeric_cols:
            scaler = StandardScaler()
            val_df[col] = scaler.fit_transform(get_values(val_df[col]))
        else:
            le = LabelEncoder()
            val_df[col] = le.fit_transform(get_values(val_df[col]))

    val_dataset = CustomDataset(val_df, transform=val_transforms, mode='inference')

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        pin_memory=True)

    model = nn.DataParallel(ClassificationModel())
    model = model.to(device)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # trained_model = r'D:\jaehwan\98.dacon\mammography\src\workspace\model\checkpoints\best_effcientnet_3_epoch_train_loss_0.656_train_acc_0.637_val_loss_0.654_val_acc_0.675.pth'
    trained_model = r'D:\jaehwan\98.dacon\mammography\src\workspace\model\checkpoints\best_effcientnet_15_epoch_train_loss_0.564_train_acc_0.848_val_loss_0.640_val_acc_0.692.pth'
    trained_model = r'D:\jaehwan\98.dacon\mammography\src\workspace\model\checkpoints\best_effcientnet_43_epoch_train_loss_0.607_train_acc_0.762_val_loss_0.631_val_acc_0.750.pth'
    model.load_state_dict(torch.load(trained_model), strict=False)

    ## TODO: INFERENCE

    preds = inference(val_loader=val_dataloader, model=model, device=device)
    preds[preds > 0.9] = 1
    preds[preds <= 0.1] = 0
    ## TODO: SUBMISSON

    submit = pd.read_csv(r'D:\jaehwan\98.dacon\mammography\csv\submit.csv')
    submit['N_category'] = preds
    submit.to_csv(r'D:\jaehwan\98.dacon\mammography\src\workspace\result\submit.csv', index=False)

    print('complete!')
