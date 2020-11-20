import cv2
import json
import numpy as np
from PIL import Image
from scipy.misc import imsave
import os

dir_with_bbox = "./results"
inference_dir = "./inference"
json_file_name = "results.json"

COLORS = np.random.uniform(0, 255, size=(11, 3))


def draw_scaled_boxes(image, boxes, labels, score, desired_size=600):
    img_size = min(image.shape[:2])
    if img_size < desired_size:
        scale_factor = float(desired_size) / img_size
    else:
        scale_factor = 1.0
    h, w = image.shape[:2]
    img_scaled = cv2.resize(image,
                            (int(w * scale_factor), int(h * scale_factor)))
    if boxes != []:
        boxes_scaled = boxes * scale_factor
        boxes_scaled = boxes_scaled.astype(np.int)
    else:
        boxes_scaled = boxes
    return draw_boxes(boxes_scaled, score, labels, img_scaled)


def draw_boxes(boxes, scores, labels, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(image, (int(box[1]), int(box[0])),
                      (int(box[3]), int(box[2])), (255, 0, 0), 3)
        cv2.putText(image, f"{labels[i]}: {scores[i]:.2f}",
                    (int(box[1]), int(box[0] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    2e-3 * image.shape[0], color, 2)
    return image


with open(json_file_name) as json_file:
    results = json.load(json_file)

for i, result in enumerate(results):
    fname = os.path.join(inference_dir, result["filename"])
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = draw_scaled_boxes(img, np.array(result["bbox"]), result["label"],
                            result["score"])
    imsave(os.path.join(dir_with_bbox, result["filename"]), img)
