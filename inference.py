"""
Check model inference.
"""
from utils.Dataset import customDataset
from utils.Loops import segmenter

import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from torchvision.models.segmentation import fcn_resnet101

from configparser import ConfigParser
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser(description='Inference for testing segmentation model')
parser.add_argument('--config', '--c', dest='config', 
    default='config.ini', type=str, help='Path to config.ini file')
args = parser.parse_args()

#~ === CONFIG ===
config = ConfigParser()
config.read(args.config)

test_path = config['INFERENCE']['inputPath']
batch_size=1
device = f"cuda:{int(config['TRAINING']['GPU'])}"
print('Using device: ', device)
torch.cuda.set_device(device)

weights_path = os.path.join(config['DIRECTORIES']['OutputDirectory'], 
        config['INFERENCE']['modelName'])


def inference(model, test_loader):
    #~ Inference loop
    id_list = []
    for idx, data in enumerate(tqdm(test_loader)):
        inputs, id_ = data['inputs'].to(
            device, dtype=torch.float32), data['id']
        id_list.append(id_)
        outputs = model(inputs)
        if idx == 0:
            input_slices = inputs
            predictions = outputs['out']
        else:
            input_slices = torch.cat((input_slices, inputs), dim=0)
            predictions = torch.cat((predictions, outputs['out']), dim=0)
    return input_slices, predictions, id_list


def main():
    #* Transforms
    test_transforms = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(
        0.229, 0.224, 0.225), max_pixel_value=1), ToTensorV2(transpose_mask=True)])

    #* Dataset
    test_dataset = customDataset(
        test_path, test_transforms, read_masks=True,
        normalise=config['TRAINING'].getboolean('Normalise'),
        window=config['TRAINING'].getint('Window'), level=config['TRAINING'].getint('Level'))
    print('Test Size', test_dataset.__len__())
    test_generator = DataLoader(test_dataset, batch_size)

    #* Load model
    model = fcn_resnet101(pretrained=False, num_classes=1).cuda()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    #~ ==== INFERENCE =====
    slices, preds, ids = inference(model, test_generator)
    np.savez(os.path.join(config['DIRECTORIES']['OutputDirectory'], 'predictions.npz'),
     slices=slices.cpu(), masks=preds.detach().cpu(), id=ids)

if __name__=='__main__':
    main()