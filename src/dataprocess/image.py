import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
import torchvision
from collections import defaultdict
import random 
from pytorch_cinic.dataset import CINIC10
from .femnist import FEMNIST

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(int(label))


def get_cinic10_as_pub(all=False, numpercls=300):
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std),])

    trainset = CINIC10(root='../data', partition='train', download=False, transform=transform_train)
    #validset = CINIC10(root='../../data', partition='valid', download=True, transform=transform_train)
    #testset = CINIC10(root='../../data', partition='test', download=True, transform=transform_train)
    print(f"the size of cinic10: {len(trainset)}")
    if all:
        return trainset
    else:
        train_labels = trainset.data.targets
        #valid_labels = validset.data.targets
       # test_labels = testset.data.targets
        train_labels = torch.Tensor(train_labels)
        #valid_labels = torch.Tensor(valid_labels)
        #test_labels = torch.Tensor(test_labels)
        indices = []
        for i in range(10):
            #indice = []
            indices += random.sample((train_labels == i).nonzero().view(-1).tolist(), numpercls)
           # indice += random.sample((valid_labels == i).nonzero().view(-1).tolist(), numpercls)
           # indice += random.sample((test_labels == i).nonzero().view(-1).tolist(), numpercls)
            #indices += indice
        publicset = Subset(trainset, indices)
        print(f"the size of publicset: {len(publicset)}")
        return publicset
    

def get_mnist_as_pub(all=False, numpercls=5000):
    trainset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))
    if all:
        print(f" the length of mnist: {len(trainset)}")
        return trainset
    else:
        labels = trainset.targets
        indices = []
        for i in range(10):
            indices += random.sample((labels == i).nonzero().view(-1).tolist(), numpercls)
        publicset = Subset(trainset, indices)
        print(f"the size of publicset: {len(publicset)}")
        return publicset

def get_cifar10_as_pub(all=False, numpercls=5000):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    if all:
        print(f" the length of mnist: {len(trainset)}")
        return trainset
    else:
        labels = torch.Tensor(trainset.targets)
        indices = []
        for i in range(10):
            indices += random.sample((labels == i).nonzero().view(-1).tolist(), numpercls)
        publicset = Subset(trainset, indices)
        print(f"the size of publicset: {len(publicset)}")
        return publicset


    





def get_femnist(share=False):
    trainset = FEMNIST('./data', train=True, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))

    testset = FEMNIST('./data', train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))


    if share:
        pri_size = int(len(trainset)*0.8) #for 10participant,set0.6
        share_size = len(trainset) - pri_size 
        new_trainset, shareset = random_split(trainset, [pri_size, share_size]) 
        print(f"train data size: {len(new_trainset)}, share data size: {len(shareset)}, test data size: {len(testset)}")
        return new_trainset, shareset, testset
    else:
        print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
        return trainset, testset






def get_cifar10(share=False):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    if share:
        pri_size = int(len(trainset)*0.6)
        share_size = len(trainset) - pri_size 
        new_trainset, shareset = random_split(trainset, [pri_size, share_size]) 
        print(f"train data size: {len(new_trainset)}, share data size: {len(shareset)}, test data size: {len(testset)}")
        return new_trainset, shareset, testset
    else:
        print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
        return trainset, testset

def get_cifar100(share=False):
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

    
    trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    
    if share:
        pri_size = int(len(trainset)*0.6)
        share_size = len(trainset) - pri_size 
        new_trainset, shareset = random_split(trainset, [pri_size, share_size]) 
        print(f"train data size: {len(new_trainset)}, share data size: {len(shareset)}, test data size: {len(testset)}")
        return new_trainset, shareset, testset
    else:
        print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
        return trainset, testset
    
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users





def cifar_noniid(dataset, no_participants, alpha=0.9):
    """
    Input: Number of participants and alpha (param for distribution)
    Output: A list of indices denoting data in CIFAR training set.
    Requires: cifar_classes, a preprocessed class-indice dictionary.
    Sample Method: take a uniformly sampled 10-dimension vector as parameters for
    dirichlet distribution to sample number of images in each class.
    """
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]

    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())
    class_size = len(cifar_classes[0])
    datasize = {}
    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            datasize[user, n] = no_imgs
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
    train_img_size = np.zeros(no_participants)
    for i in range(no_participants):
        train_img_size[i] = sum([datasize[i,j] for j in range(10)])
    clas_weight = np.zeros((no_participants,10))
    for i in range(no_participants):
        for j in range(10):
            clas_weight[i,j] = float(datasize[i,j])/float((train_img_size[i]))
    for i in per_participant_list:
        np.random.shuffle(per_participant_list[i])
    return per_participant_list, clas_weight




def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users, train_flag=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if train_flag:
        num_shards, num_imgs = 200, 300
    else:
        num_shards, num_imgs = 25, 400
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar_google_noniid(dataset, num_users, train_flag=True):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if train_flag:
        num_shards, num_imgs = 1000, 50 #200, 250
    else:
        num_shards, num_imgs = 25, 400
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 100, replace=False)) #2
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def get_mnist(share=False):
    trainset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))

    testset = datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))


    if share:
        pri_size = int(len(trainset)*0.8) #for 10participant,set0.6
        share_size = len(trainset) - pri_size 
        new_trainset, shareset = random_split(trainset, [pri_size, share_size]) 
        print(f"train data size: {len(new_trainset)}, share data size: {len(shareset)}, test data size: {len(testset)}")
        return new_trainset, shareset, testset
    else:
        print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
        return trainset, testset



def get_fmnist(share=False):
    trainset = datasets.FashionMNIST('../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))

    testset = datasets.FashionMNIST('../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))


    if share:
        pri_size = int(len(trainset)*0.8) #for 10participant,set0.6
        share_size = len(trainset) - pri_size 
        new_trainset, shareset = random_split(trainset, [pri_size, share_size]) 
        print(f"train data size: {len(new_trainset)}, share data size: {len(shareset)}, test data size: {len(testset)}")
        return new_trainset, shareset, testset
    else:
        print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
        return trainset, testset
    
    


def get_cifar100_as_pub(all=False):
    cls_ids = [0,2,20,63,71,82]
    #cls_ids = [12,14, 40,41, 42,43,44, 60, 61, 90, 91, 93]

    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),])
    trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    if all:
        return trainset
    labels = trainset.targets
    labels = torch.Tensor(labels)
    indices = []
    for i in cls_ids:
        indices += (labels == i).nonzero().view(-1).tolist()
    publicset = Subset(trainset, indices)
    print(f"train data size: {len(publicset)}")
    return publicset





class CustomSubset(Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = dataset.targets 
        self.classes = dataset.classes 

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]      
        return x, y 

    def __len__(self): 
        return len(self.indices)








def get_cifar100_with_cls(cls_use=[0,1, 7, 9, 12,18]):
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

    
    trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)

    train_labels,test_labels = torch.Tensor(trainset.targets), torch.Tensor(testset.targets)

    t_indices, v_indices = [],[]
    for i in cls_use:
        t_indices += random.sample((train_labels == i).nonzero().view(-1).tolist(), 500)
        v_indices += random.sample((test_labels == i).nonzero().view(-1).tolist(), 100)

    train_subset = CustomSubset(trainset, t_indices)
    test_subset = CustomSubset(testset, v_indices)
    print(f"the size of training subset of cifar100: {len(train_subset)}")
    print(f"the size of testing subset of cifar100: {len(test_subset)}")
    return train_subset, test_subset




class DatasetSplit4cls(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        cls = [0, 1, 7, 9, 12, 18]
        self.cls_use = {cls[i]: i + 10 for i in range(len(cls))}
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(self.cls_use[int(label)])



if __name__ == "__main__":
    #publicset = get_cifar100_as_pub()
    # publicset = get_cinic10_as_pub()
    # dataloader =DataLoader(publicset, batch_size=128,shuffle=True)
    # for i, (x,y) in enumerate(dataloader):
    #     print(type(x))
    #     print(x.size())
    trainset, testset = get_femnist()
    print(len(trainset))
    print(len(testset))
    print(dir(trainset))

