"""
Design shuffleUNet
"""
import torch
from utils.Models import Titan_base
from torchvision.models.segmentation import fcn_resnet101
from utils.Summary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    model = Titan_base().to(device)
    #model = fcn_resnet101().to(device)

    summary(model, input_size=(3,512,512))

if __name__ == '__main__':
    main()