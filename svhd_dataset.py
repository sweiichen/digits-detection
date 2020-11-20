
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd



class SVHDDataset(torch.utils.data.Dataset):
    def __init__(self, df_dir, image_dir, transforms = None):
        self.df = pd.read_csv(df_dir)
        self.image_ids = self.df['filename'].unique()
        self.image_dir = image_dir
        self.transforms = transforms
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['filename'] == image_id]
        im_name = str(image_id) + ".png"
        img = Image.open(self.image_dir+im_name).convert("RGB")
        boxes = records[['x0', 'y0', 'x1', 'y1']].values
        boxes = torch.tensor(boxes, dtype=torch.int64)
        labels = records[['label']].values
        iscrowd = torch.zeros(labels.reshape(-1,).shape, dtype=torch.int64)
        labels = torch.tensor(labels.reshape(-1,), dtype=torch.int64)
        target = {}
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["area"] = area
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['iscrowd'] = iscrowd
        if self.transforms is not None:
          img, target = self.transforms(img, target)
        return img, target
    def __len__(self):
        return self.image_ids.shape[0]