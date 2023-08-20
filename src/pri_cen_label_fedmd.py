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
from dataprocess.femnist import *
from dataprocess.my_cifar100 import *
from utils.tools import *
from utils.evaluation import *

from models.resnet import ResNet18, ResNet34
from models.vgg import *
from models.mlp import MLP
from models.cnn import *
from models.cifar10_cnn import *
from third_party.strategy import *



class Server(object):
    def __init__(self, args, share):
        self.args = args
        self.share = share
        self.loss_ce = nn.CrossEntropyLoss().cuda(self.args.gpu)
        self.loss_kl = nn.KLDivLoss(reduction='batchmean').cuda(self.args.gpu)
        loss_ce = nn.CrossEntropyLoss().cuda(self.args.gpu)
        self.loss_l2 = nn.MSELoss()

    
    def avglogit_match(self, clients, epoch):#in practice, the match operation are executed on clients
        temperature = 1
        topk_num = 3
        num_classes = 10
        if self.args.cifar100supcls:
            self.share_loader = DataLoader(DatasetSplit4cls(self.share, [i for i in range(len(self.share))]), batch_size=self.args.global_bs, shuffle=True, drop_last=True)
        else:
            self.share_loader = DataLoader(self.share, batch_size=self.args.global_bs, shuffle=True, drop_last=True)
        for e in range(self.args.distill_ep):
            for batch_idx, (images, labels) in enumerate(self.share_loader):
                images, labels = images.cuda(self.args.gpu), labels.cuda(self.args.gpu)
                if self.args.method == 'avglogits':
                    avg_logits, new_class_score_list = average_logits(clients, images, self.args)
                elif self.args.method == 'avglogits-dp':
                    avg_logits, new_class_score_list = average_dp_logits(clients, images, self.args)
                elif self.args.method == 'avglabels':
                    avg_logits, new_class_score_list = average_w_avg_labels_except_own(clients, images, self.args, 16)#16
                    #avg_logits, new_class_score_list = smooth_w_label_avg(clients, images, self.args, num_classes)

                for index in range(len(clients)):
                    loss = None
                    optimizer = torch.optim.Adam(clients[index].model.parameters())
                    if self.args.method == 'avglogits' or self.args.method == 'avglogits-dp':
                        #loss = self.loss_kl(F.log_softmax(new_class_score_list[index], dim=1), F.softmax(Variable(avg_logits)/temperature, dim=1))
                        loss = self.loss_l2(new_class_score_list[index], Variable(avg_logits))
                    elif self.args.method == 'avglabels' :
                    #loss = compute_ce_loss(self.args, new_class_score_list, index, avg_logits)
                        #loss = compute_smooth_labels_loss(self.args, new_class_score_list, index, avg_logits)
                        if self.args.cifar10cifar100 or self.args.cifar100supcls:
                            #hard_labels = compute_hard_labels(self.args, avg_logits)
                            #loss = self.loss_ce(new_class_score_list[index], hard_labels)

                            #smooth_labels = compute_smooth_labels(self.args, avg_logits)
                            smooth_labels = compute_w_smooth_labels2(self.args, avg_logits)
                            loss = compute_smooth_labels_loss(self.args, new_class_score_list, index, smooth_labels)
                            #loss = compute_smooth_labels_loss_and_KL(self.args, new_class_score_list, index, avg_logits, self.loss_kl)
                        else:
                            loss = compute_smooth_labels_loss(self.args, new_class_score_list, index, avg_logits)
                            loss += self.loss_ce(new_class_score_list[index], labels)
                    loss.backward()
                    # ce_loss = ce_loss.mean()
                    optimizer.step()
                    if args.verbose and batch_idx == (len(self.share_loader) - 1):
                        print(f"client-{index}: loss: {loss.item()}")
                #print(smooth_labels[0])
       


class Client(object):
    def __init__(self, args, dataset, share, model_type='res18'):
        self.args = args
        self.dataset = dataset
        self.share_dataset = share
        self.loss_kl = nn.KLDivLoss(reduction='batchmean').cuda(self.args.gpu)
        self.loss_ce = nn.CrossEntropyLoss().cuda(self.args.gpu)
        # load model......
        num_classes = 10 if not self.args.cifar100 else 100
        if self.args.cifar100supcls:
            num_classes = 16
        if model_type == 'res18':
            self.model = ResNet18(num_classes)
        elif model_type == 'res34':
            self.model = ResNet34(num_classes)
        elif model_type == 'vgg11':
            self.model = get_vgg11(num_classes)
        elif model_type == 'vgg16':
            self.model = get_vgg16(num_classes)
        elif model_type == 'vgg19':
            self.model = get_vgg19(num_classes)
        elif model_type == 'mlp':
            self.model = MLP(28*28*1, 200, 10)
        elif model_type == 'cnn':
            self.model = CNNCifar(num_classes)
        elif model_type == 'cifar10cnn':
            self.model = cnn_3layer_fc_model(n_classes=16)
        elif model_type == 'cnnmnist':
            #self.model = CNNMnist(16)
            self.model = EMNISTCNN(num_classes=16)
        self.model.cuda(self.args.gpu)
        #self.T = 1

    

    def _train_val_test(self, dataset, idxs, bs):
        if self.args.cifar10cnn:
            private_loader = DataLoader(DatasetSplit4cls(dataset, idxs), batch_size=bs, shuffle=True)
        else:
            private_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=bs, shuffle=True)
        return private_loader

    def assign(self, idxs):
        self.trainloader  = self._train_val_test(self.dataset, list(idxs), self.args.local_bs)
        return len(self.trainloader)

    def set_model(self, model):
        self.model = model
    
    def get_model(self):
        return self.model


    def pre_train(self, idxs, i):
        self.trainloader  = self._train_val_test(self.dataset, list(idxs), self.args.local_bs)
        if self.args.cifar100supcls:
            self.shareloader = DataLoader(DatasetSplit4cls(self.share_dataset, [i for i in range(len(self.share_dataset))]), batch_size=self.args.pre_bs, shuffle=True)
        else:
            self.shareloader = DataLoader(self.share_dataset, batch_size=self.args.pre_bs, shuffle=True)
        if bool(os.listdir(r'../fedmd_pretrain/')):
            print(f"load {i}-th pretrain model checkpoint")
            name = "test-" + '-Client-' + str(i) + '.pt'#final
            w = torch.load('../fedmd_pretrain/'+ name)
            self.model.load_state_dict(w.state_dict())
            return
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        for e in range(self.args.pre_ep):
            for loader in [self.trainloader, self.shareloader]:#, self.trainloader]:
                print("pre_train....")
                for batch_idx, (images, labels) in enumerate(loader):
                    images, labels = images.cuda(self.args.gpu), labels.cuda(self.args.gpu)
                    self.model.zero_grad()
                    log_probs = self.model(images)
                    ce_loss = self.loss_ce(log_probs, labels)
                    ce_loss.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx == (len(loader) - 1):
                        print(f"| Pre Round : {e} | Loss: {ce_loss.item()}")
        name = "test" + '-Client-' + str(i) + '.pt'#
        torch.save(self.model, '../fedmd_pretrain/' + name)


    def train(self, epoch):
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for e in range(self.args.local_ep):
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
                self.model.zero_grad()
                log_probs = self.model(images)
                ce_loss = self.loss_ce(log_probs, labels)
                ce_loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx == (len(self.trainloader) - 1):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, e, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), ce_loss.item()))

        
    

    
                



if __name__ == '__main__':
        #init...
    args = parse_args()
    setup_seed(args.seed, True if args.gpu > -1 else False)
    if args.mnist:
        data_name = 'mnist'
    elif args.mnist:
        data_name = 'fmnist'
    elif args.femnist:
        data_name = 'femnist'
    elif args.cifar10cnn:
        data_name = 'cifar10cnn'
    elif args.cifar10cifar100:
        data_name = 'cifar10cifar100'
    elif args.cifar100supcls:
        data_name = 'cifar100withsupclsaspub'
    else:
        data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    #split_type = 'googlesplit' if args.google_split else ''
    split_type = 'Dirichlet'

    
    TAG = 'FD' + data_name + '-' + split_type + '-' + str(args.noidd) + '-' + args.name + '-' + args.method + "-"
    args_str = format_args(args)
    TAG = TAG +  "-" + args_str
    print(f'{TAG}: training start....')
    
    log = Logs(TAG)
    share_flag = False
    #load dataset
    
    if args.cifar100supcls:
        train_dataset, test_dataset = get_cifar10(share_flag)
        share_dataset = get_cifar100_with_supcls_as_pub()

    elif args.femnist:#1
        train_dataset, test_dataset, user_groups = numpy2dataset(N_parties=args.num_users)
        share_dataset = get_mnist_as_pub()
    elif args.cifar10cnn:#2

        share_dataset = get_cifar10_as_pub()

        train_dataset, test_dataset, user_groups = getcifar100_with_supercls(N_parties=args.num_users)
        test_dataset = DatasetSplit4cls(test_dataset, [i for i in range(len(test_dataset))])
    else:#3
        
        train_dataset, test_dataset = get_cifar10(share_flag)
        
        share_dataset = get_cinic10_as_pub()
    
    
    if args.femnist:
        pass
    elif args.cifar10cnn:
        #user_groups = cifar_noniid_fix(train_dataset, args.num_users)
        pass
    elif args.cifar10cifar100:
        #user_groups = cifar_google_noniid(train_dataset, args.num_users)
        user_groups, _ = cifar_noniid(train_dataset, args.num_users, args.noidd)
    elif args.cifar100supcls:
        user_groups = cifar_google_noniid(train_dataset, args.num_users)
    else:
        #user_groups = cifar_iid(train_dataset, args.num_users)
        user_groups, _ = cifar_noniid(train_dataset, args.num_users, args.noidd)

    #model_types = ['cnnmnist', 'mlp']
    #model_types = ['res18', 'res34', 'vgg16']
    #model_types = ['vgg16', 'vgg16']
    
    if args.cifar10cnn:
        model_types = ['cifar10cnn', 'cifar10cnn']
    elif args.femnist:
        model_types =['cnnmnist', 'cnnmnist']
    elif args.cifar10cifar100:
        model_types = ['vgg16', 'vgg16']
    else:
        #model_types = ['vgg16', 'vgg16']
        model_types = ['vgg16', 'res18', 'vgg19']
    #model_types = ['mlp', 'mlp']
    model_list = [model_types[i % len(model_types)] for i in range(args.num_users)]
    print(model_list)
    clients = []





    acc_title = data_name + '/Acc/' 
    loss_title = data_name +  '/Loss/' 
    max_title = data_name + '/MaxAcc' 
    min_title = data_name + '/MinAcc' 
    avg_title = data_name + '/AvgAcc' 

    for i in range(len(model_list)):
        c_i = Client(args=args, dataset=train_dataset, share=share_dataset, model_type=model_list[i])
        c_i.pre_train(user_groups[i], i)
        clients.append(c_i)

        if args.cifar100:
            test_acc1, test_acc5, test_loss = test_inference4cifar100(args, c_i.get_model(), test_dataset)
            log.writer.add_scalar(acc_title + str(i), test_acc1, -1)
            log.writer.add_scalar(loss_title + str(i), test_loss, -1)
            print(f"|--Index-{i}-Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")   
        else:
            test_acc, test_loss = test_inference(args, c_i.get_model(), test_dataset)
            log.writer.add_scalar(acc_title + str(i), test_acc, -1)
            log.writer.add_scalar(loss_title + str(i), test_loss, -1)
            print(f"|--Index-{i}-Test Loss: {test_loss}, Test Accuracy: {100*test_acc}%")
    
    
    server = Server(args, share_dataset)
    
    for epoch in range(args.epochs):
        acc_dir = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        selected_clients = [clients[i] for i in idxs_users]
        if args.method != 'local':
            server.avglogit_match(selected_clients, epoch)
        #local
        for index in idxs_users:
            
            clients[index].assign(user_groups[index])
            clients[index].train(epoch)
            if args.cifar100:
                test_acc1, test_acc5, test_loss = test_inference4cifar100(args, clients[index].get_model(), test_dataset)
                acc_dir.append(test_acc1)
                log.writer.add_scalar(acc_title + str(index), test_acc1, epoch)
                log.writer.add_scalar(loss_title + str(index), test_loss, epoch)
                print(f"|--Index-{index}-Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")
            else:
                test_acc, test_loss = test_inference(args, clients[index].get_model(), test_dataset)
                acc_dir.append(test_acc)
                log.writer.add_scalar(acc_title + str(index), test_acc, epoch)
                log.writer.add_scalar(loss_title + str(index), test_loss, epoch)
                print(f"|--Index-{index}-Test Loss: {test_loss}, Test Accuracy: {100*test_acc}%")
                
        max_val, min_val, avg_val = record_test(acc_dir)
        log.writer.add_scalar(max_title, max_val, epoch)
        log.writer.add_scalar(min_title, min_val, epoch)
        log.writer.add_scalar(avg_title, avg_val, epoch)



    
