import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_std(dataset='imagenet'):
    if dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dataset == 'cityscapes':
        mean = (0.284, 0.323, 0.282)
        std = (0.175, 0.180, 0.176)
    return mean, std

def class_weight(dataset='cityscapes'):
    if dataset == 'cityscapes':
        weight = torch.FloatTensor(
                [2.8149, 6.9850, 3.7890, 9.9428, 9.7702, 9.5110, 10.3113, 10.0264,
                 4.6323, 9.5608, 7.8698, 9.5168, 10.3737, 6.6616, 10.2604, 10.2878,
                 10.2898, 10.4053, 10.13809])

    return weight