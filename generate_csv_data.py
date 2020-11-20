import h5py
import pandas as pd
from PIL import Image
import numpy as np
import os

f = h5py.File('train/digitStruct.mat', 'r')
bboxs = f['digitStruct/bbox']
names = f['digitStruct/name']


def get_img_boxes(f, annotations, idx=0):
    ann = {key: [] for key in ['height', 'left', 'top', 'width', 'label']}
    meta = {}
    box = f[bboxs[idx][0]]
    name = f[names[idx][0]]
    for key in box.keys():
        if box[key].shape[0] == 1:
            ann[key].append(float(box[key][0][0]))
        else:
            for i in range(box[key].shape[0]):
                ann[key].append(float(f[box[key][i][0]][()].item()))

    file_name = ''.join([chr(v) for v in name])
    img = Image.open('./train/' + file_name)
    meta['filename'] = os.path.splitext(file_name)[0]
    meta['width'] = img.width
    meta['height'] = img.height
    obj_count = 0
    for left, top, width, height, label in zip(ann['left'], ann['top'],
                                               ann['width'], ann['height'],
                                               ann['label']):
        meta['x0'] = left
        meta['y0'] = top
        meta['x1'] = left + width
        meta['y1'] = top + height
        meta['label'] = int(label)
        annotations = annotations.append(pd.DataFrame(meta, index=[0]))

    return annotations


annotations = pd.DataFrame(
    columns=["filename", "width", "height", "x0", "y0", "x1", "y1", "label"])

for i in range(0, 10):
    annotations = get_img_boxes(f, annotations, i)
annotations.to_csv("train/train_ann.csv", index=False)