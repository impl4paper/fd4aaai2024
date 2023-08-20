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
from models.cifar10_cnn import *
from dataprocess.femnist import *
from dataprocess.my_cifar100 import *


class Server(object):
    def __init__(self, args, share):
        self.args = args
        self.share_loader = DataLoader(share, batch_size=self.args.global_bs, shuffle=False, drop_last=True)

    
    def avglogit_match(self, clients, epoch):#in practice, the match operation are executed on clients
        temperature = 10
        loss_kl = nn.KLDivLoss(reduction='batchmean').cuda(self.args.gpu)
        loss_ce = nn.CrossEntropyLoss().cuda(self.args.gpu)
        loss_l2 = nn.MSELoss()
        mix_num = self.args.mix
        epoch_len = len(self.share_loader)
        kl_recordlist = []
        topk_num = 5
        for batch_idx, (images, labels) in enumerate(self.share_loader):
            c2logit = [[] for i in range(len(clients))]
            tmp_c2logit = [[] for i in range(len(clients))]
            new_class_score_list = []
            images, labels = images.cuda(self.args.gpu), labels.cuda(self.args.gpu)
            avg_logits = None
            eval_logits = None
            
            for c_i in range(len(clients)):
                clients[c_i].model.train()
                clients[c_i].model.zero_grad()
                class_score = clients[c_i].model(images)

                if eval_logits is None:
                    eval_logits = torch.zeros_like(class_score).cuda(self.args.gpu)
                eval_logits += class_score.clone().detach()

                _, label_index = class_score.clone().detach().topk(topk_num, 1, largest=True, sorted=False)
                label_onehot = F.one_hot(label_index, 100 if self.args.cifar100 else 10)
                multi_label = torch.sum(label_onehot, 1)
                w = [i for i in range(topk_num, 0, -1)]
                for ii in range(images.size(0)):
                    for jj in range(topk_num):
                        multi_label[ii][label_index[ii][jj].item()] = multi_label[ii][label_index[ii][jj].item()] * w[jj]

                #print(f"multi_label: {multi_label}")
                new_class_score_list.append(class_score)
                if avg_logits is None:
                    avg_logits = torch.zeros_like(multi_label).cuda(self.args.gpu)
                avg_logits += multi_label
                
            
            sum_v = torch.sum(avg_logits, 1)
            sum_v = torch.unsqueeze(sum_v, 1)
            sum_v = sum_v.repeat(1, 100 if self.args.cifar100 else 10)

            #avg_logits = avg_logits / sum_v
            eval_logits = eval_logits / len(clients)

            A = F.log_softmax(Variable(eval_logits), dim=1)
            B = F.softmax(Variable(avg_logits)/temperature, dim=1)
            kl_recordlist.append([A.view(-1).cpu().data.numpy().tolist(), B.view(-1).cpu().data.numpy().tolist()])
            #print(f"A: {A}, B: {B}")
            sim_kl = loss_kl(A, B)
            #print(f"sim_kl: {sim_kl.item()}")
            log.writer.add_scalar('Cifar10/KLDC/', sim_kl.item(), epoch_len * epoch + batch_idx)
            #log.writer.add_histogram('Cifar10/Adistributed', F.softmax(Variable(eval_logits), dim=1).clone().cpu().data.numpy(),epoch_len * epoch + batch_idx)
            #log.writer.add_histogram('Cifar10/Bdistributed', B.clone().cpu().data.numpy(),epoch_len * epoch + batch_idx)
            

            ########################
            for index in range(len(clients)):
                optimizer = torch.optim.Adam(clients[index].model.parameters())
                #print(f"new: {new_class_score_list[index]}")
                kl_loss = loss_kl(F.log_softmax(new_class_score_list[index], dim=1), F.softmax(Variable(avg_logits)/temperature, dim=1))
                #kl_loss = loss_l2(new_class_score_list[index], Variable(avg_logits_comb))
                kl_loss.backward()
                optimizer.step()

                if args.verbose and batch_idx == (len(self.share_loader) - 1):
                    print(f"client-{index}: loss: {kl_loss.item()}")
        #     if batch_idx > 100:
        #         break
        # kl_record(kl_recordlist, 'w-0.2divffinal_eval_logits_kl-' + str(topk_num) +'.csv')















    # def avglogit_match(self, clients, epoch):#in practice, the match operation are executed on clients
    #     temperature = 10
    #     loss_kl = nn.KLDivLoss(reduction='batchmean').cuda(self.args.gpu)
    #     loss_ce = nn.CrossEntropyLoss().cuda(self.args.gpu)
    #     loss_l2 = nn.MSELoss()
    #     mix_num = self.args.mix
    #     epoch_len = len(self.share_loader)
    #     kl_recordlist = []
    #     topk_num = 5
    #     for batch_idx, (images, labels) in enumerate(self.share_loader):
    #         c2logit = [[] for i in range(len(clients))]
    #         tmp_c2logit = [[] for i in range(len(clients))]
    #         new_class_score_list = []
    #         images, labels = images.cuda(self.args.gpu), labels.cuda(self.args.gpu)
    #         avg_logits = None
    #         eval_logits = None
            
    #         for c_i in range(len(clients)):
    #             clients[c_i].model.train()
    #             clients[c_i].model.zero_grad()
    #             class_score = clients[c_i].model(images)

    #             if eval_logits is None:
    #                 eval_logits = torch.zeros_like(class_score).cuda(self.args.gpu)
    #             eval_logits += class_score.clone().detach()

    #             _, label_index = class_score.clone().detach().topk(topk_num, 1, largest=True, sorted=False)
    #             label_onehot = F.one_hot(label_index, 100 if self.args.cifar100 else 10)
    #             multi_label = torch.sum(label_onehot, 1)
    #             #print(f"multi_label: {multi_label}")
    #             new_class_score_list.append(class_score)
    #             if avg_logits is None:
    #                 avg_logits = torch.zeros_like(multi_label).cuda(self.args.gpu)
    #             avg_logits += multi_label
                
            
    #         sum_v = torch.sum(avg_logits, 1)
    #         sum_v = torch.unsqueeze(sum_v, 1)
    #         sum_v = sum_v.repeat(1, 100 if self.args.cifar100 else 10)

    #         #avg_logits = avg_logits / sum_v
    #         eval_logits = eval_logits / len(clients)

    #         A = F.log_softmax(Variable(eval_logits), dim=1)
    #         B = F.softmax(Variable(avg_logits)/temperature, dim=1)
    #         #kl_recordlist.append([A.view(-1).cpu().data.numpy().tolist(), B.view(-1).cpu().data.numpy().tolist()])
    #         #print(f"A: {A}, B: {B}")
    #         sim_kl = loss_kl(A, B)
    #         #print(f"sim_kl: {sim_kl.item()}")
    #         log.writer.add_scalar('Cifar10/KLDC/', sim_kl.item(), epoch_len * epoch + batch_idx)
    #         #log.writer.add_histogram('Cifar10/Adistributed', F.softmax(Variable(eval_logits), dim=1).clone().cpu().data.numpy(),epoch_len * epoch + batch_idx)
    #         #log.writer.add_histogram('Cifar10/Bdistributed', B.clone().cpu().data.numpy(),epoch_len * epoch + batch_idx)
            

    #         ########################
    #         for index in range(len(clients)):
    #             optimizer = torch.optim.Adam(clients[index].model.parameters())
    #             #print(f"new: {new_class_score_list[index]}")
    #             kl_loss = loss_kl(F.log_softmax(new_class_score_list[index], dim=1), F.softmax(Variable(avg_logits)/temperature, dim=1))
    #             #kl_loss = loss_l2(new_class_score_list[index], Variable(avg_logits_comb))
    #             kl_loss.backward()
    #             optimizer.step()

    #             if args.verbose and batch_idx == (len(self.share_loader) - 1):
    #                 print(f"client-{index}: loss: {kl_loss.item()}")
    #         # if batch_idx > 100:
    #         #     break
    #     #kl_record(kl_recordlist, '10nodivffinal_eval_logits_kl-' + str(topk_num) +'.csv')

            
            




            

            








        
    

if __name__ == '__main__':
        #init...
    args = parse_args()
    setup_seed(args.seed, True if args.gpu > -1 else False)
    if args.mnist:
        data_name = 'mnist'
    elif args.mnist:
        data_name = 'fmnist'
    elif args.cifar10cnn:
        data_name = 'cifar10cnn'
    else:
        data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    #split_type = 'googlesplit' if args.google_split else ''
    split_type = 'dilikelei'
    
    TAG = 'global_model_training' + data_name + '-' + split_type + '-' + args.name 
    args_str = format_args(args)
    TAG = TAG +  "-" + args_str
    print(f'{TAG}: training start....')
    
    log = Logs(TAG)
    share_flag = False
    #load dataset
    if args.cifar100:
        train_dataset, share_dataset, test_dataset = get_cifar100(share_flag)
    elif args.mnist:
        train_dataset, share_dataset, test_dataset = get_mnist(share_flag)
    elif args.fmnist:
        train_dataset,share_dataset, test_dataset = get_fmnist(share_flag)
    elif args.cifar10cnn:
        share_dataset = get_cifar10_as_pub()
        train_dataset, test_dataset, user_groups = getcifar100_with_supercls(N_parties=args.num_users)
        #train_dataset = DatasetSplit4cls(train_dataset, [i for i in range(len(train_dataset))])
        test_dataset = DatasetSplit4cls(test_dataset, [i for i in range(len(test_dataset))])
    else:
        #train_dataset, share_dataset, test_dataset = get_cifar10(share_flag)
        #train_dataset, test_dataset = get_cifar10(share_flag)
        #share_dataset = get_cifar100_as_pub()
        train_dataset, test_dataset = get_cifar10(share_flag)
        #share_dataset = get_cifar100_as_pub()
        share_dataset = get_cinic10_as_pub()
    #     user_groups, _ = cifar_noniid(train_dataset, args.num_users, args.noidd)
    # loaders = []
    # for i in range(args.num_users):
    #     loaders.append(DataLoader(DatasetSplit(train_dataset, user_groups[i]), batch_size=args.local_bs, shuffle=True))
    # loaders.append(DataLoader(share_dataset, batch_size=args.global_bs, shuffle=True))
    num_classes = 10 if not args.cifar100 else 100
    model = get_vgg16(num_classes)
    #model = cnn_3layer_fc_model(n_classes=16)
    model.cuda(args.gpu)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    ce = nn.CrossEntropyLoss().cuda(args.gpu)
    loaders = []
    loaders.append(DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True))
    loaders.append(DataLoader(share_dataset, batch_size=args.global_bs, shuffle=True))
    for epoch in range(args.epochs):
        for trainloader in loaders:
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
                model.zero_grad()
                log_probs = model(images)
                ce_loss = ce(log_probs, labels)
                ce_loss.backward()
                optimizer.step()

                if args.verbose and batch_idx == (len(trainloader) - 1):
                    print('| Global Round : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(images),
                        len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), ce_loss.item()))
        if args.cifar100:
            test_acc1, test_acc5, test_loss = test_inference4cifar100(args, model, test_dataset)
            log.writer.add_scalar('Cifar100/Acc/-1', test_acc1, epoch)
            log.writer.add_scalar('Cifar100/Loss/-1', test_loss, epoch)
            print(f"|Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")
                
        else:
            test_acc, test_loss = test_inference(args, model, test_dataset)
            log.writer.add_scalar('Cifar10/Acc/-1', test_acc, epoch)
            log.writer.add_scalar('Cifar10/Loss/-1', test_loss, epoch)
            print(f"|Test Loss: {test_loss}, Test Accuracy: {100*test_acc}%")


    torch.save(model.state_dict(), '../globalmodel/' + TAG  + '.pt')        
    



    

