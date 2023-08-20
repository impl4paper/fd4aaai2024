import os
import pandas as pd
from datetime import datetime
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy

def test_error(model, steal_model, test_dataset):
    model.eval()
    steal_model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.cuda(0), labels.cuda(0)

            # Inference
            outputs0 = model(images)
            outputs1 = steal_model(images)
            
            # Prediction
            _, pred_labels0 = torch.max(outputs0, 1)
            pred_labels0 = pred_labels0.view(-1)
            
            _, pred_labels1 = torch.max(outputs1, 1)
            pred_labels1 = pred_labels1.view(-1)
            
            correct += torch.sum(torch.eq(pred_labels0, pred_labels1)).item()
            total += len(labels)
    accuracy = correct/total
    return accuracy
    

"""
for single training on cifar10
"""

def test_inference(args, net, test_dataset):
    """ Returns the test accuracy and loss.
    """
    net.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        criterion.cuda(args.gpu)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            # Inference
            outputs = net(images)
            batch_loss = criterion(outputs, labels)
            loss += copy.deepcopy(batch_loss.item())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    accuracy = correct/total
    loss = loss /total
    return accuracy, loss

"""
for single training in cifar100
"""
def test_inference4cifar100(args, net, test_dataset):
    """ Returns the test accuracy and loss.
    """
    net.eval()
    loss, total, correct_1, correct_5 = 0.0, 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        criterion.cuda(args.gpu)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            # Inference
            outputs = net(images)
            batch_loss = criterion(outputs, labels)
            loss += copy.deepcopy(batch_loss.item())

            # Prediction
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1 
            correct_1 += correct[:, :1].sum()

    accuracy_1 = correct_1 / len(testloader.dataset)
    accuracy_5 = correct_5 / len(testloader.dataset)
    return accuracy_1, accuracy_5, loss