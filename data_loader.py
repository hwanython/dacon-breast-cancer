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
        patient_id = self.states.iloc[:, 0].values[idx]
        img_path = self.states.iloc[:, 1].values[idx]
        label = self.states.iloc[:, 2].values[idx]


        img = np.load(img_path)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        if self.mode == 'train':
            # img = torch.from_numpy(img).float()
            # label = torch.from_numpy(label).long()
            label = np.array([label])
            return img, label
        else:
            return img, img_path