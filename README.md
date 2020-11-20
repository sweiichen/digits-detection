# Digits Detection
HW2 for NCTU CS Selected Topics in Visual Recognition using Deep Learning
The task is to detect every single digits in a image, and use SVHN dataset whcich contains 33402 training images, 13068 test images to do the experiment.

I reference from the [pytorch offical tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) and use the pretrained faster R-CNN model to implement this task.
The used python code, [utils.py](https://github.com/sweiichen/digits-detection/blob/main/utils.py), [transforms.py](https://github.com/sweiichen/digits-detection/blob/main/transforms.py), [coco_eval.py](https://github.com/sweiichen/digits-detection/blob/main/coco_eval.py), [engine.py](https://github.com/sweiichen/digits-detection/blob/main/engine.py) and [coco_utils.py](https://github.com/sweiichen/digits-detection/blob/main/coco_utils.py), are from https://github.com/pytorch/vision.git.



## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- GeForce GTX 1080 Ti

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Inference](#inference)

## Installation
All requirements should be detailed in [requirements.txt](https://github.com/sweiichen/digits-detection/blob/main/requirements.txt). 
You have to create your own python virtual environment.
- python version: 3.7.7 
- cuda version: 10.1.243

```
pip install -r requirements.txt
```

## Dataset Preparation
Download the whole training dataset from [svhd](http://ufldl.stanford.edu/housenumbers/).
### Annotation processing 
After downloding it, we have to deal with the digitalStruct.mat file which containing the annotations of the training datas.
Just run generate_csv_data.py and get the train_ann.csv file to fit the [custom dataset](https://github.com/sweiichen/digits-detection/blob/main/svhd_dataset.py) which I write.
If you don't want to run this file, can just use [train_ann.csv](https://github.com/sweiichen/digits-detection/blob/main/train/train_ann.csv) saved in [./train](https://github.com/sweiichen/digits-detection/tree/main/train) folder.

```
python generate_csv_data.py
```




## Training
Save all the training images and train_ann.csv in the [./train](https://github.com/sweiichen/digits-detection/tree/main/train) folder and run [train.py](https://github.com/sweiichen/digits-detection/blob/main/train.py) to start the training process. 

```
python train.py
```
I split the train data to traing set and validataion set with 90-10 ratio.
You might chage the bath size, according to you GPU memory size.
The expected training times are:

 GPUs  | Epoch | Training Time
------------ | ------------- | ------------- 
 1x TitanX  | 1 | 1 hours 10 mins

When starting running the code, you can see the ouput like this.
```
Epoch: [0]  [   0/7516]  eta: 1:39:30  lr: 0.000010  loss: 2.0864 (2.0864)  loss_classifier: 1.8552 (1.8552)  loss_box_reg: 0.1045 (0.1045)  loss_objectness: 0.1105 (0.1105)  loss_rpn_box_reg: 0.0162 (0.0162)  time: 0.7944  data: 0.1111  max mem: 0
Epoch: [0]  [1000/7516]  eta: 1:02:11  lr: 0.005000  loss: 0.2485 (0.3681)  loss_classifier: 0.1323 (0.2315)  loss_box_reg: 0.1030 (0.1148)  loss_objectness: 0.0022 (0.0082)  loss_rpn_box_reg: 0.0117 (0.0135)  time: 0.5750  data: 0.0050  max mem: 0
Epoch: [0]  [2000/7516]  eta: 0:52:45  lr: 0.005000  loss: 0.1819 (0.2951)  loss_classifier: 0.1019 (0.1784)  loss_box_reg: 0.0655 (0.0963)  loss_objectness: 0.0025 (0.0072)  loss_rpn_box_reg: 0.0104 (0.0132)  time: 0.5805  data: 0.0050  max mem: 0
Epoch: [0]  [3000/7516]  eta: 0:43:12  lr: 0.005000  loss: 0.1647 (0.2604)  loss_classifier: 0.0933 (0.1539)  loss_box_reg: 0.0550 (0.0878)  loss_objectness: 0.0010 (0.0061)  loss_rpn_box_reg: 0.0094 (0.0127)  time: 0.5677  data: 0.0050  max mem: 0
.
.
Epoch: [0] Total time: 1:11:56 (0.5743 s / it)
creating index...
index created!
Test:  [  0/836]  eta: 0:17:31  model_time: 0.2450 (0.2450)  evaluator_time: 0.0162 (0.0162)  time: 1.2574  data: 0.0986  max mem: 0
Test:  [100/836]  eta: 0:03:11  model_time: 0.2304 (0.2317)  evaluator_time: 0.0119 (0.0121)  time: 0.2508  data: 0.0054  max mem: 0
Test:  [200/836]  eta: 0:02:42  model_time: 0.2285 (0.2327)  evaluator_time: 0.0127 (0.0122)  time: 0.2521  data: 0.0052  max mem: 0
.
.
```

The validation output after one epoch:
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.439
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.886
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.363
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.427
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.487
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.576
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
```
After 1 epochs I obtain a COCO-style mAP of 43.9.

After training for 10 epochs, I got the following metrics and the mAP is 48.3.
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.483
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.932
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.437
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.523
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.647
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
```



### Load trained parameters
I have save the trained model parameters in the [google drive](https://drive.google.com/file/d/1YlV7PhROIFigH4E-lgHxWlUN6bkRWeOA/view?usp=sharing), you could directly download it and save in ./model folder to do inference without retraining.


## Inference
The inference time per image is about 122 ms when using GPU from google Colab.
### Inference and get the bbox information
You should put the test images in the [./inference](https://github.com/sweiichen/digits-detection/tree/main/inference) folder and run [inference.py](https://github.com/sweiichen/digits-detection/blob/main/inference.py) and you will get the results information in the results.json.
```
python inference.py
```
The json format is a list of dictionaries.
Each dictionary contains four keys
- "filename": the file name of a image 
- "bbox": list of bounding boxes in (y1, x1, y2, x2)
- "score": list of probability for the class
- "label": list of label
which looks like: 
```
{
    "filename": "1.png",
    "bbox": [[7, 40, 40, 60]],
    "score": [0.9948711395263672], 
    "label": [5]
}
```

### Draw bounding box
After getting the results.json, you can run [draw_bbox.py](https://github.com/sweiichen/digits-detection/blob/main/draw_bbox.py) and use the json file to get the images with predicted bounding boxes in the [./results](https://github.com/sweiichen/digits-detection/tree/main/results) folder
```
python draw_bbox.py
```

![](https://i.imgur.com/N7OwQAc.png) 
![](https://i.imgur.com/aVjS6Ww.png)





