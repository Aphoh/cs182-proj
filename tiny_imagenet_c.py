import torch
import torchvision
import numpy as np

all_corruptions = ['gaussian_noise',
    'glass_blur',
    'impulse_noise',
    'frost',
    'shot_noise',
    'motion_blur',
    'defocus_blur',
    'elastic_transform',
    'brightness',
    'zoom_blur',
    'fog',
    'contrast',
    'snow',
    'pixelate',
    'jpeg_compression']

def get_tiny_imagenet_c(transform, severity=1, corruptions=all_corruptions,
                        reduce_size=True, data_dir="./data/Tiny-ImageNet-C"):
    datasets = []
    for corruption in corruptions:
        dataset = torchvision.datasets.ImageFolder(data_dir+'/'+corruption+'/'+str(severity), transform)
        datasets.append(dataset)
    if reduce_size:
        N = len(datasets[0])
        K = len(corruptions)
        indices = np.array_split(np.arange(N, dtype=int), K)
        datasets = [torch.utils.data.Subset(datasets[i], indices[i]) for i in range(K)]
    return torch.utils.data.ConcatDataset(datasets)
