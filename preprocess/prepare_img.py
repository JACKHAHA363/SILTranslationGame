import torch
from itlearn.img_utils import MyImageFolder, preprocess_10rc, preprocess_1c, preprocess_5c, preprocess_10c
from tqdm import tqdm
import sys


def _extract_img_features(cfg):
    model = eval('models.%s(pretrained=True)' % cfg.model_name)
    model = torch.nn.Sequential(*list(model.children())[:-1])
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
