import torch.utils.data.dataset
import torchvision
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json
import _init_paths
import dataset
from tqdm import tqdm
import numpy as np
import pickle
import time
import math
import cv2
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


# VOCSegmentation dataset을 정의합니다.
class myVOCSegmentation(VOCSegmentation):
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            augmented = self.transforms(image=np.array(img), mask=np.array(target))
            img = augmented['image']
            target = augmented['mask']
            target[target > 20] = 0

        semantic_map = []
        for colour in range(21):
            equality = np.equal(target, colour)
            semantic_map.append(equality)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)

        img = to_tensor(img)
        target = torch.from_numpy(semantic_map).type(torch.float32).permute(2, 0, 1)
        return img, target


def re_normalize(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    x_r = x.clone()
    for c, (mean_c, std_c) in enumerate(zip(mean, std)):
        x_r[c] *= std_c
        x_r[c] += mean_c
    return x_r
