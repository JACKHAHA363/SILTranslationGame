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
from torchvision.datasets.folder import has_file_allowed_extension, default_loader, IMG_EXTENSIONS
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


class _DatasetFolder(datasets.DatasetFolder):
    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(datasets.DatasetFolder, self).__init__(root, transform=transform,
                                                     target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def make_dataset(self, class_to_idx, extensions=None, is_valid_file=None):
        images = []
        root = os.path.expanduser(self.root)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return has_file_allowed_extension(x, extensions)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(root, target)
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                path = os.path.join(d, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
        return images


class ImageFolderWithPaths(_DatasetFolder):
    # override the __getitem__ method. this is the method dataloader calls

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolderWithPaths, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                             transform=transform,
                                             target_transform=target_transform,
                                             is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
