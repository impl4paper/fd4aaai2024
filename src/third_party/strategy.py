import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable




def average_logits(clients, images, args):
    avg_logits = None
    new_class_score_list = []
    for c_i in range(len(clients)):
        clients[c_i].model.train()
        clients[c_i].model.zero_grad()
        class_score = clients[c_i].model(images)
        new_class_score_list.append(class_score)
        if avg_logits is None:
            avg_logits = torch.zeros_like(class_score).cuda(args.gpu)
        avg_logits += class_score.clone().detach()
    avg_logits = avg_logits / len(clients)
    return avg_logits, new_class_score_list



def average_dp_logits(clients, images, args):
    avg_logits = None
    new_class_score_list = []
    laplace = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
    for c_i in range(len(clients)):
        clients[c_i].model.train()
        clients[c_i].model.zero_grad()
        class_score = clients[c_i].model(images)
        nb, nc = class_score.size(0), class_score.size(1)
        noise = laplace.sample((nb,nc)).to(args.gpu).squeeze(dim=-1)
        class_score += noise
        new_class_score_list.append(class_score)
        if avg_logits is None:
            avg_logits = torch.zeros_like(class_score).cuda(args.gpu)
        avg_logits += class_score.clone().detach()
    avg_logits = avg_logits / len(clients)
    return avg_logits, new_class_score_list




def average_labels(clients, images, args):
    avg_logits = None
    topk_num = 5
    new_class_score_list = []
    for c_i in range(len(clients)):
        clients[c_i].model.train()
        clients[c_i].model.zero_grad()
        class_score = clients[c_i].model(images)
        #_, label_index = class_score.clone().detach().topk(topk_num, 1, largest=True, sorted=False)
        _, label_index = F.softmax(class_score.clone().detach(), dim=1).topk(topk_num, 1, largest=True, sorted=False)
        label_onehot = F.one_hot(label_index, 100 if args.cifar100 else 10)
        multi_label = torch.sum(label_onehot, 1).float()
        new_class_score_list.append(class_score)
        if avg_logits is None:
            avg_logits = torch.zeros_like(multi_label).cuda(args.gpu)
        avg_logits += multi_label
    avg_logits = avg_logits / len(clients)# fix avg
    return avg_logits, new_class_score_list



def average_w_avg_labels(clients, images, args):
    avg_logits = None
    new_class_score_list = []
    topk_num = 5
    for c_i in range(len(clients)):
        clients[c_i].model.train()
        clients[c_i].model.zero_grad()
        class_score = clients[c_i].model(images)
        label_value, label_index = F.softmax(class_score.clone().detach(), dim=1).topk(topk_num, 1, largest=True, sorted=True)
        label_onehot = F.one_hot(label_index, 100 if args.cifar100 else 10)
        multi_label = torch.sum(label_onehot, 1).float()
        max_val = label_value[:, 0]
        max_val_ = max_val.repeat(100 if args.cifar100 else 5, 1)
        max_val_t = max_val_.transpose(0, 1)
        w = label_value / max_val_t
        for ii in range(images.size(0)):
            for jj in range(topk_num):
                multi_label[ii][label_index[ii][jj].item()] = multi_label[ii][label_index[ii][jj].item()] * w[ii][jj]
        new_class_score_list.append(class_score)
        if avg_logits is None:
            avg_logits = torch.zeros_like(multi_label).cuda(args.gpu)
        avg_logits += multi_label
    avg_logits = avg_logits / len(clients)# fix avg
    return avg_logits, new_class_score_list


def average_w_avg_labels_except_own(clients, images, args, num_classes):
    avg_logits = None
    new_class_score_list = []
    topk_num = args.labelnum
    for c_i in range(len(clients)):
        clients[c_i].model.train()
        clients[c_i].model.zero_grad()
        class_score = clients[c_i].model(images)
        label_value, label_index = F.softmax(class_score.clone().detach(), dim=1).topk(topk_num, 1, largest=True, sorted=True)
        label_onehot = F.one_hot(label_index, num_classes)
        multi_label = torch.sum(label_onehot, 1).float()
        max_val = label_value[:, 0]
        max_val_ = max_val.repeat(topk_num, 1)
        max_val_t = max_val_.transpose(0, 1)
        w = label_value / max_val_t
        w = F.softmax(Variable(w))
        for ii in range(images.size(0)):
            for jj in range(topk_num):
                multi_label[ii][label_index[ii][jj].item()] = multi_label[ii][label_index[ii][jj].item()] * w[ii][jj].item()
        new_class_score_list.append(class_score)
        if avg_logits is None:
            avg_logits = torch.zeros_like(multi_label).cuda(args.gpu)
        avg_logits += multi_label
    avg_logits = avg_logits / len(clients)# fix avg
    return avg_logits, new_class_score_list







def smooth_w_label_avg(clients, images, args, num_classes):
    avg_logits = None
    new_class_score_list = []
    topk_num = args.labelnum
    for c_i in range(len(clients)):
        clients[c_i].model.train()
        clients[c_i].model.zero_grad()
        class_score = clients[c_i].model(images)
        new_class_score_list.append(class_score)
        score_clone = class_score.clone().detach()
        score_f = F.softmax(score_clone, dim=1)
        _, label_index = score_f.topk(topk_num, 1, largest=True, sorted=False)
        label_onehot = F.one_hot(label_index, num_classes)
        multi_label = torch.sum(label_onehot, 1).float()
        
        temp_val, _ = score_f.topk(1, 1, largest=True, sorted=False)
        w = score_f / temp_val
        w_multi_label = multi_label * w
        if avg_logits is None:
            avg_logits = torch.zeros_like(w_multi_label).cuda(args.gpu)
        avg_logits += w_multi_label

    avg_logits = avg_logits / len(clients)# fix avg
    return avg_logits, new_class_score_list





def compute_hard_labels(args, avg_logits):
    _, pred_labels = torch.max(avg_logits, 1)
    pred_labels = pred_labels.view(-1)
    return pred_labels


def compute_smooth_labels(args, avg_logits):
    num_classes = 10
    _, ind = torch.max(avg_logits, 1)
    label_onehot = F.one_hot(ind, num_classes)
    alpha = 0.1
    temp = torch.zeros_like(avg_logits).fill_(1/num_classes).to(args.gpu)
    smooth_labels = (1 - alpha) * label_onehot + alpha * temp
    smooth_labels = Variable(smooth_labels)
    return smooth_labels



def compute_w_smooth_labels(args, avg_logits):
    num_classes = 10
    _, ind = torch.max(avg_logits, 1)
    label_onehot = F.one_hot(ind, num_classes)
    alpha = 0.1

    temp = avg_logits.clone().detach()
    ind_new = ind.view(-1, 1)
    temp.scatter_(1, ind_new, 0)
    #smooth_labels = (1 - alpha) * label_onehot + alpha * temp
    #smooth_labels =  label_onehot + temp  #no
    smooth_labels = (1 - alpha) * label_onehot + alpha * temp 
    smooth_labels = Variable(smooth_labels)
    return smooth_labels


def compute_w_smooth_labels2(args, avg_logits):
    num_classes = 10
    alpha = 0.1

    temp = avg_logits.clone().detach()
    temp2 = torch.zeros_like(avg_logits).fill_(1/num_classes).to(args.gpu)
    
    smooth_labels = (1 - alpha) * temp + alpha * temp2
    smooth_labels = Variable(smooth_labels)
    return smooth_labels















def compute_ce_loss(args, new_class_score_list, index, avg_logits, num_classes):
    loss_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    loss = None
    #for label_id in range(100 if args.cifar100 else 10):
    for label_id in range(num_classes):
        pred_label = torch.zeros(new_class_score_list[index].size(0)).fill_(label_id).long().cuda(args.gpu)
        if loss is None:
            loss = avg_logits[:, label_id] * loss_ce(new_class_score_list[index], pred_label)
        else:
            loss += avg_logits[:, label_id] * loss_ce(new_class_score_list[index], pred_label)
    
    return loss.mean()

def compute_smooth_labels_loss(args, new_class_score_list, index, avg_logits):
    loss = None
    pred = new_class_score_list[index]
    pred = F.log_softmax(pred, -1)
    loss = torch.mean(torch.sum(-Variable(avg_logits) * pred, dim=-1))
    return loss


def compute_smooth_labels_and_ce_loss(args, new_class_score_list, index, avg_logits):
    loss = None
    pred = new_class_score_list[index]
    pred = F.log_softmax(pred, -1)
    return torch.mean(torch.sum(-Variable(avg_logits) * pred, dim=-1))


def compute_smooth_labels_loss_and_KL(args, new_class_score_list, index, avg_logits, KL):
    temperature = 1
    loss = None
    pred = new_class_score_list[index]
    pred = F.log_softmax(pred, -1)
    loss = torch.mean(torch.sum(-Variable(avg_logits) * pred, dim=-1))
    loss += KL(F.log_softmax(new_class_score_list[index], dim=1), F.softmax(Variable(avg_logits)/temperature, dim=1))
    return loss