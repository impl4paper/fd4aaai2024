import os
import copy
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
import argparse
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision

from utils.options import parse_args
from dataprocess.image import *
from utils.tools import *
from utils.evaluation import *

from models.resnet import ResNet18, ResNet34
from models.vgg import get_vgg16

from models.mlp import MLP
from models.cnn import CNNCifar, CNNMnist

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class Client(object):
    def __init__(self, args, dataset, model_type='res18'):
        self.args = args
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss().cuda(self.args.gpu)
        # load model......
        num_classes = 10 if not self.args.cifar100 else 100
        if model_type == 'res18':
            self.model = ResNet18(num_classes)
        elif model_type == 'res34':
            self.model = ResNet34(num_classes)
        elif model_type == 'vgg16':
            self.model = get_vgg16(num_classes)
        elif model_type == 'mlp':
            self.model = MLP(32*32*3, 200, num_classes)
        elif model_type == 'cnn':
            self.model = CNNCifar(num_classes)
        elif model_type == 'cnnmnist':
            self.model = CNNMnist(10)
        if self.args.gpu > -1:
            self.model.cuda(self.args.gpu)

    def _train_val_test(self, dataset, idxs):
        trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=self.args.local_bs, shuffle=True)
        return trainloader, None, None

    def assign(self, idxs):
        self.trainloader, self.validloader, self.testloader = self._train_val_test(self.dataset, list(idxs))
        return len(self.trainloader)

    def get_model(self):
        return self.model

    def set_model(self, w):
        self.model.load_state_dict(w)

    def train(self, epoch):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        for e in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx == (len(self.trainloader) - 1):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, e, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))

        



if __name__ == '__main__':
    #init...
    args = parse_args()
    setup_seed(args.seed, True if args.gpu > -1 else False)
    
    if args.mnist:
        data_name = 'mnist'
    else:
        data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    #split_type = 'googlesplit' if args.google_split else ''
    split_type = 'dilikelei'
    
    TAG = 'localtraining-' + data_name + '-' + split_type + '-' + args.name 
    args_str = format_args(args)
    TAG = TAG +  "-" + args_str
    print(f'{TAG}: training start....')
    
    log = Logs(TAG)
    
    #load dataset
    if args.cifar100:
        train_dataset, test_dataset = get_cifar100()
    elif args.mnist:
        train_dataset, test_dataset = get_mnist()
    else:
        train_dataset, test_dataset = get_cifar10()
    #user_groups, _ = cifar_noniid(train_dataset, args.num_users * 2)
    # user_groups = list(user_groups.values())
    # global_group = user_groups[args.num_users:]
    # user_groups = user_groups[0: args.num_users]
    if args.mnist:
        user_groups = mnist_noniid(train_dataset, args.num_users)
        #test_user_groups = mnist_noniid(test_dataset, args.num_users, False)

    else:
        user_groups, _ = cifar_noniid(train_dataset, args.num_users)
    #user_groups = cifar_iid(train_dataset, args.num_users)


     #init clients 
    #model_types = ['res18', 'res34', 'vgg16']
    #model_types = ['vgg16', 'vgg16']
    model_types = ['cnnmnist', 'cnnmnist']
    model_list = [model_types[i % len(model_types)] for i in range(args.num_users)]
    print(model_list)
    clients = []
    for i in range(len(model_list)):
        clients.append(Client(args=args, dataset=train_dataset, model_type=model_list[i]))

    

    for epoch in range(args.epochs):
        for index in range(len(clients)):
            print(f'Epoch: {epoch}-id:{index}==============================')
            clients[index].assign(user_groups[index])
            clients[index].train(epoch)

        
            if args.cifar100:
                test_acc1, test_acc5, test_loss = test_inference4cifar100(args, clients[index].get_model(), test_dataset)
                log.writer.add_scalar('test/CIFAR100_Acc/' + str(index), test_acc1, epoch)
                log.writer.add_scalar('test/CIFAR100_Loss/' + str(index), test_loss, epoch)
                print(f"|--Index-{index}-Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")
            else:
                #test_acc, test_loss = test_inference(args, clients[index].get_model(), DatasetSplit(test_dataset, test_user_groups[index]))
                test_acc, test_loss = test_inference(args, clients[index].get_model(), test_dataset)
                log.writer.add_scalar('test/Mnist_Acc/' + str(index), test_acc, epoch)
                log.writer.add_scalar('test/Mnist_Loss/' + str(index), test_loss, epoch)
                print(f"|--Index-{index}-Test Loss: {test_loss}, Test Accuracy: {100*test_acc}%")
       

            


