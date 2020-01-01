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


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]  # return image path

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


def _extract_img_features(cfg):
    model = eval('models.%s(pretrained=True)' % cfg.model_name)
    model = nn.Sequential(*list(model.children())[:-1])
    model = torch.nn.DataParallel(model).cuda()

    assert cfg.crops in [1, 5, 10]

    if cfg.rand_crop:
        dataset = MyImageFolder(cfg.root, preprocess_10rc)
    elif cfg.crops == 1:
        dataset = MyImageFolder(cfg.root, preprocess_1c)
    elif cfg.crops == 5:
        dataset = MyImageFolder(cfg.root, preprocess_5c)
    elif cfg.crops == 10:
        dataset = MyImageFolder(cfg.root, preprocess_10c)

    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4,
                                         pin_memory=True)
    print('Extracting %s image features...' % cfg.model_name)
    features = {}
    progress = tqdm(loader, mininterval=1, leave=False, file=sys.stdout)
    for (inputs, target), files in progress:
        if cfg.rand_crop:
            bs, n_crops, c, h, w = inputs.size()
            outputs = model(Variable(inputs.view(bs * n_crops, c, h, w)))
            outputs = normf(outputs.view(bs * n_crops, -1))
            outputs = outputs.view(bs, n_crops, -1).data
        elif cfg.crops == 1:
            outputs = model(Variable(inputs)).data
            outputs = normf(outputs)
        elif cfg.crops in [5, 10]:
            bs, n_crops, c, h, w = inputs.size()
            outputs = model(Variable(inputs.view(bs * n_crops, c, h, w)))
            outputs = normf(outputs.view(bs * n_crops, -1))
            outputs = outputs.view(bs, n_crops, -1).mean(1).data
            outputs = normf(outputs)

        for ii, img_path in enumerate(files[0]):
            img_name = img_path.split('/')[-1]
            if cfg.rand_crop:
                features[img_name] = [outputs[ii, c, :].squeeze().cpu().numpy() for c in range(n_crops)]
            else:
                features[img_name] = outputs[ii, :].squeeze().cpu().numpy()

    return features


if __name__ == '__main__':
    # Extract image features
    import argparse

    # Flickr30 images features
    cfg = argparse.Namespace()
    cfg.rand_crop = True
    cfg.crops = 1
    cfg.batch_size = 6
    cfg.root = '/network/tmp1/luyuchen/cv_datasets/flickr30k-images'
    cfg.model_name = 'resnet152'
    features = _extract_img_features(cfg)
    torch.save(features, '')
