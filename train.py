import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import torchvision
from engine import train_one_epoch, evaluate
import utils
from svhd_dataset import SVHDDataset
from model_utils import get_detection_model, get_transform
from datetime import datetime

train_dir = "train/"
train_annotations = "train/train_ann.csv"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_classes = 11 # 10 class (digits) + background

#use defined dataset to get data
dataset = SVHDDataset(train_annotations,train_dir,get_transform(train=True))
dataset_test = SVHDDataset(train_annotations,train_dir,get_transform(train=False))
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:int(len(indices)*0.9)])
dataset_test = torch.utils.data.Subset(dataset_test, indices[int(len(indices)*0.9):])
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=4, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)

model = get_detection_model(num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
num_epochs = 11

for epoch in range(num_epochs):    
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1000)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)   

torch.save({"epoch": 10,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "lr_scheduler":lr_scheduler.state_dict()},
           f"model/faster_rcnn_{datetime.now().month}{datetime.now().day}.pth")                                           

