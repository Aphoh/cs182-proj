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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Net
from torch.utils.data import Dataset
from torch import nn
from PIL import Image
import pandas as pd
import vgg_slim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

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

def get_n_params(model):
        pp=0
        for name, param in model.named_parameters():
            if param.requires_grad:
                nn=1
                for s in list(param.size()):
                    nn = nn*s
                pp += nn
        return pp


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    model_name = 'experiment_ricap_VGG16_slim_baseline'
    writer = SummaryWriter('runs/' + model_name)
    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = 32
    im_height = 64
    im_width = 64
    num_epochs = 120
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2296, 0.2263, 0.2255]
    ricap_beta = 0.3

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(), 
      transforms.ToTensor(), 
      transforms.Normalize(mean, std)])
    preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean, std)])

    train_set_init = torchvision.datasets.ImageFolder(data_dir / 'train', train_transform)
    #train_set = AugMixDataset(train_set_init, preprocess,augmentations.augmentations_all)
    train_loader = torch.utils.data.DataLoader(train_set_init, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    val_set = Val_Dataset(val_transforms,train_set_init.class_to_idx)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               shuffle=False, num_workers=4, pin_memory=True)


    # Create a simple modeli

    lr = 0.001
    model = vgg_slim.vgg16_slim().cuda()
    optim = torch.optim.SGD(model.parameters(),lr=lr, momentum=.9, weight_decay=5e-4)
    sched = ReduceLROnPlateau(optim, 'min', patience=3, factor=0.2, verbose=True)

    print(get_n_params(model))
    criterion = nn.CrossEntropyLoss()
    for i in range(num_epochs):
        model.train()

        train_total, train_correct = 0,0
        running_loss = 0.0
        for idx_outer, (inputs, targets) in enumerate(train_loader):

        	### RICAP START ###
            I_x, I_y = inputs.size()[2:]

            w = int(np.round(I_x * np.random.beta(ricap_beta, ricap_beta)))
            h = int(np.round(I_y * np.random.beta(ricap_beta, ricap_beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]

            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                idx = torch.randperm(inputs.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = inputs[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                c_[k] = targets[idx].cuda()
                W_[k] = w_[k] * h_[k] / (I_x * I_y)

            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3)
            patched_images = patched_images.cuda()

            output = model(patched_images)
            loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])

            acc = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])
            acc = acc.item()

            ### RICAP END ###

            running_loss += loss.item()
            loss.backward()
            optim.step()

            print("\r", end='')
            print(f'training {100 * idx_outer / len(train_loader):.2f}%: {acc:.3f}', end='')


        torch.save({
            'net': model.state_dict(),
        }, './models/' + model_name +'.pt')

        writer.add_scalar('Train Accuracy', float(train_correct)/float(train_total),i)
        writer.add_scalar('Train Loss', running_loss, i)

        model.eval()
        val_total, val_correct = 0,0
        running_loss = 0.0
        for idx, (inputs, targets) in enumerate(val_loader):

            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            print("\r", end='')
            print(f'validation {100 * idx / len(val_loader):.2f}%: {val_correct / val_total:.3f}', end='')

        writer.add_scalar('Validation Accuracy', float(val_correct)/float(val_total), i)
        writer.add_scalar('Validation Loss', running_loss, i)

        sched.step(running_loss)

    writer.close()


if __name__ == '__main__':
    main()
