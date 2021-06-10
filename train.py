"""
Main Training script 
"""
from albumentations.augmentations.transforms import ElasticTransform
import torch
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.optim import optimizer

from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim


from utils.customDataset import customDataset
from utils.loops import segmenter
from utils.customWriter import customWriter
import cv2

train_path = './data/training/'
valid_path = './data/testing/'

batch_size = 4
num_epochs=500
learning_rate = 3e-4

weight_path = '/home/donal/data/Donal_Backup/ResNet/COCO/experiments/bootstrap_132/iteration_1/repeat_0/best_model.pt'


def load_weights(pt_model):
    model = models.segmentation.fcn_resnet101(pretrained=False, num_classes=1)
    #* Load weights
    pt_dict = torch.load(pt_model)
    model_dict = model.state_dict()
    pretrained_dict = {}
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
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(
            0.229, 0.224, 0.225), max_pixel_value=1),
        ToTensor()
    ])
    #* Normalise to ImageNet mean and std 
    valid_transforms = A.Compose(
        [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1), ToTensor()])

    train_dataset = customDataset(train_path, train_transforms, read_masks=True)
    valid_dataset = customDataset(
        valid_path, valid_transforms, read_masks=True)
    train_loader = DataLoader(train_dataset, batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size)

    model = load_weights(weight_path)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = customWriter(batch_size)


    seg = segmenter(model, optimizer, train_loader, valid_loader, writer, num_epochs, device='cuda:3')
    seg.forward()


if __name__ == '__main__':
    main()
