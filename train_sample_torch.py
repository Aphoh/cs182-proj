"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""
import sys
sys.path.append('EfficientNet-PyTorch')
from efficientnet_pytorch import EfficientNet
import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import Net

from torch import nn
def main():
    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = 512
    im_height = 64
    im_width = 64
    num_epochs = 120

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_set = torchvision.datasets.ImageFolder(data_dir / 'val', data_transforms)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    # Create a simple modeli


    model = EfficientNet.from_name('efficientnet-b0').cuda()
    criterion = nn.CrossEntropyLoss()
    lr = 1e-2
    for i in range(num_epochs):
        model.train()
        if(i%40==0):
            lr *= .1
        optim = torch.optim.Adam(model.parameters(),lr=lr)
        train_total, train_correct = 0,0
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            print("\r", end='')
            print(f'training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}', end='')
        torch.save({
            'net': model.state_dict(),
        }, 'latest.pt')
        model.eval()
        val_total, val_correct = 0,0
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            print("\r", end='')
            print(f'validation {100 * idx / len(val_loader):.2f}%: {val_correct / val_total:.3f}', end='')



if __name__ == '__main__':
    main()
