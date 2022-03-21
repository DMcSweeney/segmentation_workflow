"""
Main Training script 
"""
from albumentations.augmentations.geometric.transforms import ElasticTransform
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import optimizer

from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim

from configparser import ConfigParser
from argparse import ArgumentParser

from utils.Dataset import customDataset
from utils.Loops import segmenter
from utils.Writer import customWriter
import cv2
import os

#~ Arg. Parse for config file
parser = ArgumentParser(description='Inference for testing segmentation model')
parser.add_argument('--config', '--c', dest='config',
                    default='config.ini', type=str, help='Path to config.ini file')
args = parser.parse_args()


#~ ====  Read config === 
config  = ConfigParser()
config.read('config_cbct.ini')

root_dir = config['DIRECTORIES']['InputDirectory']

train_path = os.path.join(root_dir, 'split_data/training')
valid_path = os.path.join(root_dir, 'split_data/testing')

#~ PARAMS
batch_size = int(config['TRAINING']['BatchSize'])
num_epochs= int(config['TRAINING']['NumEpochs'])
learning_rate = float(config['TRAINING']['LR'])
device = f"cuda:{int(config['TRAINING']['GPU'])}"
print('Using device: ', device)
torch.cuda.set_device(device)
weight_path = config['TRAINING']['InitWeights']
input_size = int(config['TRAINING']['InputSize'])

def load_weights(pt_model):
    model = models.segmentation.fcn_resnet101(pretrained=False, num_classes=1)
    #* Load weights
    pt_dict = torch.load(pt_model, map_location=device)
    model_dict = model.state_dict()

    pretrained_dict = {}
    #~ In case loading from different architecture
    for key, val in pt_dict.items():
        if key in model_dict:
            if val.shape == model_dict[key].shape:
                pretrained_dict[key] = val
            else:
                print("Shapes don't match")
                continue
        else:
            print("Key not in dict")
            continue
    # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # Load new state dict
    model.load_state_dict(model_dict)
    return model

def main():
    #* Define augmentations
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5, limit=20, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.GridDistortion(p=0.5, num_steps=3, distort_limit=0.3, border_mode=4, interpolation=1),
        A.RandomScale(),
        A.Resize(input_size, input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(
            0.229, 0.224, 0.225), max_pixel_value=1),
        ToTensorV2(transpose_mask=True)
    ])
    #* Normalise to ImageNet mean and std 
    valid_transforms = A.Compose(
        [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
        max_pixel_value=1), 
        ToTensorV2(transpose_mask=True)])

    #~ init. Datasets
    train_dataset = customDataset(train_path, train_transforms, read_masks=True, 
            normalise=config['TRAINING'].getboolean('Normalise'),
            window=config['TRAINING'].getint('Window'), level=config['TRAINING'].getint('Level'))

    valid_dataset = customDataset(
        valid_path, valid_transforms, read_masks=True, 
        normalise = config['TRAINING'].getboolean('Normalise'),
        window=config['TRAINING'].getint('Window'), level=config['TRAINING'].getint('Level'))

    train_loader = DataLoader(train_dataset, batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size)

    model = load_weights(weight_path)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = customWriter(batch_size)

    #~ ==== TRAIN =====
    seg = segmenter(model, optimizer, train_loader,
                    valid_loader, writer, num_epochs, 
                    device=device, output_path=config['DIRECTORIES']['OutputDirectory']
                    )
    seg.forward()


if __name__ == '__main__':
    main()
