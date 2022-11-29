import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, df, mode='train', transform=None):
        self.states = df
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_path = self.states['img_path'].values[idx]
            label = self.states['N_category'].values[idx]
            tabular = torch.Tensor(
                self.states.drop(columns=['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']).iloc[idx])
        else:
            img_path = self.states['img_path'].values[idx]
            tabular = torch.Tensor(
                self.states.drop(columns=['ID', 'img_path', '수술연월일']).iloc[idx])

        img = np.load(img_path)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        if self.mode == 'train':
            label = np.array([label])
            return img, label, tabular
        else:
            return img, tabular
