
from PIL import Image
import os.path
import torch
import scipy.io as sio
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import random



class CIFAR100Coarse(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, superclass=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)
        if superclass:
        # update labels
            coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
            self.targets = coarse_labels[self.targets]
            self.targets = self.targets.tolist()

            
            
            
fine_classes_in_use = [[4, 32, 6, 68, 64, 58],
 [72, 1, 7, 37, 66, 8],
 [55, 67, 14, 76, 34, 13],
 [30, 73, 18, 12, 75, 48],
 [95, 91, 24, 17, 63, 90],
 [4, 32, 6, 68, 64, 58],
 [72, 1, 7, 37, 66, 8],
 [55, 67, 14, 76, 34, 13],
 [30, 73, 18, 12, 75, 48],
 [95, 91, 24, 17, 63, 90]]


relations = [[4, 72, 55, 30, 95],
 [32, 1, 67, 73, 91],
 [70, 82, 54, 92, 62],
 [9, 10, 16, 28, 61],
 [0, 83, 51, 53, 57],
 [39, 40, 22, 87, 86],
 [5, 84, 20, 25, 94],
 [6, 7, 14, 18, 24],
 [97, 3, 42, 43, 88],
 [68, 37, 76, 12, 17],
 [33, 71, 49, 23, 60],
 [38, 15, 19, 21, 31],
 [64, 66, 34, 75, 63],
 [99, 77, 45, 79, 26],
 [2, 35, 98, 11, 46],
 [44, 78, 29, 27, 93],
 [65, 36, 74, 80, 50],
 [96, 47, 52, 56, 59],
 [58, 8, 13, 48, 90],
 [69, 41, 81, 85, 89]]

private_classes =[0,1, 7, 9, 12,18]#超类

def getcifar100_with_supercls(N_parties=10):
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),])
    supercls_train = CIFAR100Coarse(root='../data', train=True, download=True, transform=transform_train, superclass=True)
    subcls_train = CIFAR100Coarse(root='../data', train=True, download=True, transform=transform_train, superclass=False)
    supercls_test = CIFAR100Coarse(root='../data', train=False, download=True, transform=transform_test, superclass=True)
    
    user_dict = {i: [] for i in range(N_parties)}
    subcls_train_labels = torch.Tensor(subcls_train.targets)
    for index, cls_list in enumerate(fine_classes_in_use):
        t_indices = []
        for i in cls_list:
            t_indices += random.sample((subcls_train_labels == i).nonzero().view(-1).tolist(), 20)
        user_dict[index] += t_indices

    v_indices = []
    supercls_test_labels = torch.Tensor(supercls_test.targets)
    for cls in private_classes:
        v_indices += random.sample((supercls_test_labels == cls).nonzero().view(-1).tolist(), 500)
    testsubset = Subset(supercls_test, v_indices)
    return supercls_train, testsubset, user_dict
    


def get_cifar100_with_supcls_as_pub(all=False):
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),])

    supercls_train = CIFAR100Coarse(root='../data', train=True, download=True, transform=transform_train, superclass=True)
    labels = torch.Tensor(supercls_train.targets)
    indices = []
    for i in private_classes:
        indices += random.sample((labels == i).nonzero().view(-1).tolist(), 500)
    
    return Subset(supercls_train, indices)
        
    


