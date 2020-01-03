import torch
from itlearn.img_utils import ImageFolderWithPaths, preprocess_rc, normf
from tqdm import tqdm
import sys
import argparse
import os
import torch
import torchvision.models as models


def _extract_img_features(cfg):
    model = eval('models.%s(pretrained=True)' % cfg.model_name)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = torch.nn.DataParallel(model).cuda()
    dataset = ImageFolderWithPaths(cfg.root, preprocess_rc)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4,
                                         pin_memory=True)
    print('Extracting %s image features...' % cfg.model_name)
    features = {}
    progress = tqdm(loader, mininterval=1, leave=False, file=sys.stdout)
    for batch in progress:
        inputs, _, files = batch
        outputs = model(inputs).data
        outputs = normf(outputs)
        for ii, img_path in enumerate(files):
            img_name = os.path.basename(img_path)
            features[img_name] = outputs[ii, :].squeeze().cpu().numpy()
    return features


def get_args():
    # Extract image features
    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', required=True)
    args = parser.parse_args()
    datadir = args.datadir
    return datadir


def main():
    data_dir = get_args()
    flickr30k_dir = os.path.join(data_dir, 'flickr30k')

    # Flickr30 images features
    if not os.path.exists(os.path.join(flickr30k_dir, 'train_feat.pth')):
        cfg = argparse.Namespace()
        cfg.rand_crop = True
        cfg.crops = 1
        cfg.batch_size = 64
        cfg.root = flickr30k_dir
        cfg.model_name = 'resnet152'
        feats = _extract_img_features(cfg)

        with open(os.path.join(flickr30k_dir, 'train.txt')) as f:
            train_split = f.readlines()
            train_split = [line.rstrip('\n') for line in train_split]
            train_feats = [feats[fname] for fname in train_split]
        with open(os.path.join(flickr30k_dir, 'val.txt')) as f:
            val_split = f.readlines()
            val_split = [line.rstrip('\n') for line in val_split]
            val_feats = [feats[fname] for fname in val_split]
        torch.save(train_feats, os.path.join(flickr30k_dir, 'train_feat.pth'))
        torch.save(val_feats, os.path.join(flickr30k_dir, 'val_feat.pth'))


if __name__ == '__main__':
    main()
