"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""
import sys
sys.path.append('augmix')
from datasets import AugMixDataset
import augmentations
sys.path.append('EfficientNet-PyTorch')

from efficientnet_pytorch import EfficientNet
import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import Net
from torch.utils.data import Dataset
from torch import nn
from PIL import Image
import pandas as pd

class Val_Dataset(Dataset):
    def __init__(self, transform,indexer, anns='./data/tiny-imagenet-200/val/val_annotations.txt', path='./data/tiny-imagenet-200/val/images/',):
        self.path = path
        self.anns = pd.read_csv(anns, sep = '\t', header= None)
        
        self.indexer = indexer
        self.transform = transform
        self.samples = []
        self.anns[1] = self.anns[1].map(indexer)

        for index, row in self.anns.iterrows():
            
            self.samples.append((path + self.anns.iloc[index][0], self.anns.iloc[index][1]))



    def loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def __getitem__(self,index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
        
            
    def __len__(self):
        return len(self.anns)


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

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])

    train_set = torchvision.datasets.ImageFolder(data_dir / 'train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_set = Val_Dataset(val_transforms,train_set.class_to_idx)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               shuffle=False, num_workers=4, pin_memory=True)

    # Create a simple model

    


    model = EfficientNet.from_name('efficientnet-b0').cuda()
    def get_n_params(model):
        pp=0
        for name, param in model.named_parameters():
            if param.requires_grad:
                nn=1
                for s in list(param.size()):
                    nn = nn*s
                pp += nn
        return pp
    print(get_n_params(model))
    ckpt = torch.load('latest.pt')
    model.load_state_dict(ckpt['net'])
    criterion = nn.CrossEntropyLoss()
    lr = 5e-2
    for i in range(num_epochs):
        model.train()
        if(i%40==0):
            lr *= .1
        optim = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9   )
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
