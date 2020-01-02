# Copyright 2017-present Facebook. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torchvision.transforms as T
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as models
import torchvision.datasets as datasets
from torch.autograd import Variable
from tqdm import tqdm
import os
import sys


def normf(t, p=2, d=1):
    return t / t.norm(p, d, keepdim=True).expand_as(t)


def get_pil_img(root, img_name):
    img_pil = Image.open(os.path.join(root, img_name))
    # UNTESTED, but there are grayscale images (e.g. COCO_train2014_000000549879.jpg)
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    return img_pil


# imagenet
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

preprocess_1c = T.Compose([
    T.Resize(size=256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean, std)
])

preprocess_5c = T.Compose([
    T.Resize(size=256),
    T.FiveCrop(224),
    T.Lambda(lambda crops: [T.ToTensor()(c) for c in crops]),
    T.Lambda(lambda crops: [T.Normalize(mean, std)(c) for c in crops]),
    T.Lambda(lambda crops: torch.stack(crops))
])

preprocess_10c = T.Compose([
    T.Resize(size=256),
    T.TenCrop(224),
    T.Lambda(lambda crops: [T.ToTensor()(c) for c in crops]),
    T.Lambda(lambda crops: [T.Normalize(mean, std)(c) for c in crops]),
    T.Lambda(lambda crops: torch.stack(crops))
])

preprocess_rc = T.Compose([
    T.RandomResizedCrop(size=224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean, std)
])


preprocess_10rc = T.Compose([
    T.Lambda(lambda im: [T.RandomResizedCrop(224)(im) for _ in range(10)]),
    T.Lambda(lambda crops: [T.RandomHorizontalFlip()(c) for c in crops]),
    T.Lambda(lambda crops: [T.ToTensor()(c) for c in crops]),
    T.Lambda(lambda crops: [T.Normalize(mean, std)(c) for c in crops]),
    T.Lambda(lambda crops: torch.stack(crops))
])


class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
