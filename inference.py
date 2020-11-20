import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import torchvision
import matplotlib.pyplot as plt
from model_utils import get_detection_model, get_transform
import json

dir_root = "./inference"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_classes = 11
file_name = 'results.json' 
checkpoint = torch.load('model/faster_rcnn_1118.pth')

model = get_detection_model(num_classes)
model = model.to(device)
model.load_state_dict(checkpoint["model_state_dict"])
results = []
model.eval()
threshold = 0.5

for file in os.listdir(dir_root):
    image = Image.open(os.path.join(dir_root,file))
    img = torchvision.transforms.ToTensor()(image)
    
    with torch.no_grad():
        prediction = model([img.to(device)])
    for pred in prediction:
        bboxes = []
        scores = []
        labels = []
        for bbox, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            if score > threshold:
                bbox = list(bbox.cpu().numpy().astype(int))
                #bbox (y0, x0, y1, x1)
                bbox = [int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]
                bboxes.append(bbox)
                scores.append(float(score.cpu().numpy()))
                labels.append(int(label.cpu().numpy()))
        p = dict(
            filename=file,
            bbox=bboxes,
            score=scores,
            label=labels
        )
    results.append(p)

with open(file_name,'w') as f:
    json.dump(results,f)