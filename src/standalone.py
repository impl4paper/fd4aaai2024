import os
import copy
import time
import pickle
import math
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
from models.vgg import *
from models.mlp import MLP
from models.cnn import *



class Client(object):
    def __init__(self, args, dataset, pub, model_type='res18'):
        self.args = args
        self.dataset = dataset
        self.public_dataset = pub
        self.loss_kl = nn.KLDivLoss(reduction='batchmean').cuda(self.args.gpu)
        self.loss_ce = nn.CrossEntropyLoss().cuda(self.args.gpu)
        # load model......
        num_classes = 10 if not self.args.cifar100 else 100
        if model_type == 'res18':
            self.model = ResNet18(num_classes)
        elif model_type == 'res34':
            self.model = ResNet34(num_classes)
        elif model_type == 'vgg16':
            self.model = get_vgg16(num_classes)
        elif model_type == 'vgg19':
            self.model = get_vgg19(num_classes)
        elif model_type == 'mlp':
            self.model = MLP(28*28*1, 200, num_classes)
        elif model_type == 'cnn':
            self.model = CNNCifar(num_classes)
        elif model_type == 'cnnmnist':
            self.model = CNNMnist(10)
        self.model.cuda(self.args.gpu)

    def _train_val_test(self, dataset, idxs):
        trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=self.args.local_bs, shuffle=True)
        publicloader = DataLoader(self.public_dataset, batch_size=self.args.local_bs, shuffle=True)
        return trainloader, publicloader

    def assign(self, idxs):
        
        self.trainloader, self.publicloader = self._train_val_test(self.dataset, idxs)
        
        return len(self.trainloader)

    def set_model(self, model):
        self.model = model
    
    def get_model(self):
        return self.model


    def train(self, epoch):

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        for e in range(1):
            for loader in [self.trainloader]: 
                for batch_idx, (images, labels) in enumerate(loader):
                    images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
                    self.model.zero_grad()
                    pred_probs = self.model(images)
                    loss = self.loss_ce(pred_probs, labels)                                                                       
                    loss.backward()
                    optimizer.step()
            
                    if self.args.verbose and batch_idx == (len(loader) - 1):
                        print('| First：Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, e, batch_idx * len(images),
                            len(loader.dataset),
                            100. * batch_idx / len(loader), loss.item()))

        


if __name__ == '__main__':
        #init...
    args = parse_args()
    setup_seed(args.seed, True if args.gpu > -1 else False)
    if args.mnist:
        data_name = 'mnist'
    elif args.fmnist:
        data_name = 'fmnist'
    else:
        data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    #split_type = 'googlesplit' if args.google_split else ''
    split_type = 'dilikelei'
    
    TAG = 'standalone' + data_name + '-' + split_type + '-' + args.name 
    args_str = format_args(args)
    TAG = TAG +  "-" + args_str
    print(f'{TAG}: training start....')
    
    log = Logs(TAG)
    
    #load dataset
    if args.cifar100:
        train_dataset, public_dataset, test_dataset = get_cifar100(True)
    elif args.mnist:
        train_dataset, public_dataset,test_dataset = get_mnist(True)
    elif args.fmnist:
        train_dataset, public_dataset,test_dataset = get_fmnist(True)
    else:
        train_dataset, public_dataset, test_dataset = get_cifar10(True)
    
    if args.mnist or args.fmnist:
        #user_groups =  mnist_iid(train_dataset, args.num_users)
        #user_groups =  mnist_noniid(train_dataset, args.num_users)
        user_groups, _ = cifar_noniid(train_dataset, args.num_users)
    else:
        user_groups, _ = cifar_noniid(train_dataset, args.num_users)
        #user_groups = cifar_iid(train_dataset, args.num_users)
        
        

    #model_types = ['vgg19', 'vgg19']
    #model_types = ['res18', 'res34', 'vgg16']
    model_types = ['cnnmnist','cnnmnist']
    #model_types = ['res18', 'res18']
    #model_types = ['vgg16','vgg16']
    model_list = [model_types[i % len(model_types)] for i in range(args.num_users)]
    print(model_list)
    clients = []
    for i in range(len(model_list)):
        clients.append(Client(args=args, dataset=train_dataset, pub=public_dataset, model_type=model_list[i]))
   
    #init.....
    acc_dir = {i: [] for i in range(args.num_users)}
    for epoch in range(args.epochs):
        for v in range(len(clients)):
            clients[v].assign(user_groups[v])
            clients[v].train(epoch)
            if args.cifar100:
                test_acc1, test_acc5, test_loss = test_inference4cifar100(args, clients[v].get_model(), test_dataset)
                log.writer.add_scalar('Cifar100/Acc/' + str(v), test_acc1, epoch)
                log.writer.add_scalar('Cifar100/Loss/' + str(v), test_loss, epoch)
                print(f"|--Index-{v}-Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")
                
            else:
                test_acc, test_loss = test_inference(args, clients[v].get_model(), test_dataset)
                #log.writer.add_scalar('MNIST/Acc/' + str(v), test_acc, epoch)
                #log.writer.add_scalar('MNIST/Loss/' + str(v), test_loss, epoch)
                acc_dir[v].append(test_acc)
                print(f"|--Index-{v}-Test Loss: {test_loss}, Test Accuracy: {100*test_acc}%")


    record_max(acc_dir)
                



    

