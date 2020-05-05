import sys
sys.path.append('EfficientNet-PyTorch')
from efficientnet_pytorch import EfficientNet
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

"""
Mean: tensor([0.4802, 0.4481, 0.3975])
Stdev: tensor([0.2296, 0.2263, 0.2255])
"""

def main():
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    data_transforms = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)

    loader = DataLoader(train_set, batch_size=10, num_workers=1, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data,b in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Mean: {}".format(mean))
    print("Stdev: {}".format(std))

if __name__ == "__main__":
    main()
