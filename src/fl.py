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
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision

from utils.options import parse_args
from dataprocess.image import *
from utils.tools import *
from utils.evaluation import *

from models.resnet import ResNet18, ResNet34
from models.vgg import get_vgg16



def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class Client(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss().cuda(self.args.gpu)

    def _train_val_test(self, dataset, idxs):
        trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=self.args.local_bs, shuffle=True)
        return trainloader, None, None

    def assign(self, idxs):
        self.trainloader, self.validloader, self.testloader = self._train_val_test(self.dataset, list(idxs))
        return len(self.trainloader)

    def train(self, model, epoch):
        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        for e in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx == (len(self.trainloader) - 1):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, e, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
        return model.state_dict()

        



if __name__ == '__main__':
    #init...
    args = parse_args()
    setup_seed(args.seed, True if args.gpu > -1 else False)

    data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    num_classes = 10 if not args.cifar100 else 100
    #split_type = 'googlesplit' if args.google_split else ''
    split_type = 'dilikelei'
    TAG = 'fedavg-' + data_name + '-' + split_type + '-' + args.name 
    print(f'{TAG}: training start....')
    
    log = Logs(TAG)
    #load dataset
    if args.cifar100:
        train_dataset, test_dataset = get_cifar100()
    else:
        train_dataset, test_dataset = get_cifar10()
    #user_groups, _ = cifar_noniid(train_dataset, args.num_users)
    user_groups = cifar_iid(train_dataset, args.num_users)
    

    # load model......
    if args.model == 'res18':
        global_model = ResNet18(num_classes)
        model_str = 'res18'
    elif args.model == 'res34':
        global_model = ResNet34(num_classes)
        model_str = 'res34'
    elif args.model == 'vgg16':
        global_model = get_vgg16(num_classes)
        model_str = 'vgg16'
    if args.gpu > -1:
        global_model.cuda(args.gpu)

    args_str = format_args(args)
    
    TAG = TAG + model_str + "-" + args_str

    # simulate clients
    local_model = Client(args=args, dataset=train_dataset)
    
    
    for epoch in range(args.epochs):
        
        local_weights = []
        print(f'epoch: {epoch}================================================')
        global_model.train()
        global_weights = global_model.state_dict()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            print(f'Epoch: {epoch}-id:{idx}==============================')
            local_model.assign(user_groups[idx])
            local_weights.append(local_model.train(copy.deepcopy(global_model), epoch))
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        if args.cifar100:
            test_acc1, test_acc5, test_loss = test_inference4cifar100(args, , test_dataset)
            log.writer.add_scalar('test/CIFAR100_Acc/' + str(idx), test_acc1, epoch)
            log.writer.add_scalar('test/CIFAR100_Loss/' + str(idx), test_loss, epoch)
            print(f"|--Index-{idx}-Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")
        else:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            log.writer.add_scalar('test/CIFAR10_Acc/' + str(idx), test_acc, epoch)
            log.writer.add_scalar('test/CIFAR10_Loss/' + str(idx), test_loss, epoch)
            print(f"|--Index-{idx}-Test Loss: {test_loss}, Test Accuracy: {100*test_acc}%")
        

    #     if args.cifar100:
    #         test_acc1, test_acc5, test_loss = test_inference4cifar100(args, global_model, test_dataset)
    #         print(f"|----Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")
    #         log_obj = {
    #             'test_acc1': "{:.2f}%".format(100*test_acc1),
    #             'test_acc5': "{:.2f}%".format(100*test_acc5),
    #             'loss': test_loss,
    #             'epoch': epoch 
    #             }
    #         logs.append(log_obj)
    #     else:
    #         test_acc, test_loss = test_inference(args, global_model, test_dataset)
    #         print(f"|----Test Loss: {test_loss}, Test Accuracy: {100*test_acc}%")
    #         log_obj = {
    #             'test_acc': "{:.2f}%".format(100*test_acc),
    #             'loss': test_loss,
    #             'epoch': epoch 
    #             }
    #         logs.append(log_obj)
    # if args.cifar100:
    #     save_cifar100_logs(logs, TAG, args)
    # else:
    #     save_logs(logs, TAG,  args)

            


