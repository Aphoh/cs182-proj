import torch
import torchvision
import torchvision.transforms as transforms
import os
import pathlib
import numpy as np
import pandas as pd

import vgg_slim
from torch.utils.data import Dataset
from torch import nn
from PIL import Image
import torchvision.datasets as dsets
from torch import optim
from augmix_vgg_train import Val_Dataset

def create_model(model_name):

    model = None
    if model_name == 'vgg16_slim':
        model = vgg_slim.vgg16_slim().cuda()
    elif model_name == 'vgg16':
        model = vgg_slim.vgg16().cuda()
    elif model_name == 'efficientnet-b0':
        model = EfficientNet.from_name('efficientnet-b0').cuda()
    elif model_name == 'resnet-18':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False).cuda()

    else:
        print("Unknown model name: {}".format(model_name))

    return model

def get_val(model):
    # Create a pytorch dataset
    data_dir = pathlib.Path(args.datadir) # SET THIS TO THE DATADIR
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = args.batch_size
    im_height = 64
    im_width = 64
    num_epochs = args.epochs
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2296, 0.2263, 0.2255]

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip()])
    preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean, std)])

    train_set_init = torchvision.datasets.ImageFolder(data_dir / 'train', train_transform)
    train_loader = torch.utils.data.DataLoader(train_set_init, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    val_set = Val_Dataset(val_transforms,train_set_init.class_to_idx)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               shuffle=False, num_workers=4, pin_memory=True)


    criterion = nn.CrossEntropyLoss()
    
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
	return val_correct / val_total


def add_votes(model, weight):
    # Create a pytorch dataset
    data_dir = pathlib.Path(args.datadir) # SET THIS TO THE DATADIR
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = args.batch_size
    im_height = 64
    im_width = 64
    num_epochs = args.epochs
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2296, 0.2263, 0.2255]

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip()])
    preprocess = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize(mean, std)])

    train_set_init = torchvision.datasets.ImageFolder(data_dir / 'train', train_transform)
    train_loader = torch.utils.data.DataLoader(train_set_init, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    val_set = Val_Dataset(val_transforms,train_set_init.class_to_idx)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               shuffle=False, num_workers=4, pin_memory=True)


    criterion = nn.CrossEntropyLoss()
    
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

	return val_correct / val_total


# try the fibonacci numbers up to like 55 for good results 
def alpha_priority(names, models, accuracies, new_model, alpha=0):
    model_dict = {}
    for i in range(len(names)):
        model_dict[names[i]] = {'acc':accuracies[i], 'params': models[i].named_parameters() }


    entry = 'alpha_{}'.format(alpha)
    weighted_model_params = {}
    for name, _ in models[0].named_parameters():
        weighted_model_params[name] = 0
    total_val = 0
    for k, v in model_dict.items():
        total_val += v['acc'] ** alpha
    for k, v in model_dict.items():
        v[entry] = (v['acc'] ** alpha) / total_val
        for name, param in v['params']:
            weighted_model_params[name] += v[entry] * param
    new_model.load_state_dict(weighted_model_params)
    return model_dict, new_model


def main():

	model_dir = './models'
        model_dict = {}
	model_type = 'vgg16-slim'


	names = []
	models = []
	accs = []
	for model_name in [f for f in os.listdir(model_dir) if 'experiment_augmix_vgg16_slim_checkpoint_' in f]:
            print('evaluating model ' + str(model_name))

	    model = create_model(model_type)
            model.eval()
            models.append(model)
            names.append(model_name)
	    pretrained_dict = torch.load(model_dir + model_name)
	    weight_dict = model.state_dict()
	    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in weight_dict}
	    weight_dict.update(pretrained_dict)
	    model.load_state_dict(weight_dict)

	    #model.load_state_dict(torch.load(model_dir + model_name))
	    accuracy = get_val(model)
	    accs.append(accuracy)
	    model_dict[model_name] = {'params': model.named_parameters(), 'acc': accuracy}


	res = {}
	outputs = None
	for alpha in range(5):
		new_model = create_model(model_type)

		# PARAMETER AVERAGING
		model_dict, model = alpha_priority(names, models, accs, new_model, alpha)
		res['param_avg_alpha_'.format(alpha)]: get_val(model)

		# # VOTING
		# for k, v in model_dict.items():
		# 	weight = v['alpha_'.format(alpha)]

	print(res)
	return res


if __name__ == '__main__':
    main()
