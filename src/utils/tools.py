import numpy as np
import random
import torch
import time 
import os
import sys
import pandas as pd 
from tensorboardX import SummaryWriter

import pandas as pd
from datetime import datetime
import time

def setup_seed(seed=777, gpu_enabled=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if gpu_enabled:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True





def format_args(args):
    return "frac{}-bs{}-users{}-epochs{}".format(args.frac, args.local_bs, args.num_users, args.epochs)



class Logs(object):
    def __init__(self, name):
        root = r'../heto'
        if not os.path.exists(root):
            os.mkdir(root)
        self.writer = SummaryWriter(os.path.join(root, name))
    def close(self):
        self.writer.close()


def save_logs(logs, tag, args):
    df = pd.DataFrame(logs)
    param_str = format_args(args)
    path = '../logs/{}_{}_{}.csv'.format(tag, param_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, mode='a',index_label='index')
    df['test_acc'] = df['test_acc'].apply(lambda x: float(x.replace('%', '')))
    print(f"final Accuracy: {df.loc[:,'test_acc'].max()}")
    print("save logs sucess!")


def save_cifar100_logs(logs, tag, args):
    df = pd.DataFrame(logs)
    param_str = format_args(args)
    path = '../logs/{}_{}_{}.csv'.format(tag, param_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, mode='a',index_label='index')

    df['test_acc1'] = df['test_acc1'].apply(lambda x: float(x.replace('%', '')))
    print(f"final Accuracy1: {df.loc[:,'test_acc1'].max()}")
    df['test_acc5'] = df['test_acc5'].apply(lambda x: float(x.replace('%', '')))
    print(f"final Accuracy5: {df.loc[:,'test_acc5'].max()}")
    print("save logs sucess!")


def record_max(acc_dir):
    best_acc = []
    for k, v in acc_dir.items():
        best_acc.append(max(v))
    max_val, min_val, avg_val = max(best_acc), min(best_acc), sum(best_acc)/len(best_acc)
    print(f"max: {max_val}, min: {min_val}, avg: {avg_val}")
    return max_val, min_val, avg_val
    #return max(best_acc), min(best_acc), sum(best_acc)/len(best_acc)

def record_test(acc_dir):
    max_val, min_val, avg_val = max(acc_dir), min(acc_dir), sum(acc_dir)/len(acc_dir)
    print(f"max: {max_val}, min: {min_val}, avg: {avg_val}")
    return max_val, min_val, avg_val


def kl_record(kl_list, path):
    df = pd.DataFrame(kl_list)
    #param_str = format_args(args)
    #path = '../kl_logs/{}_{}_{}.csv'.format(tag, param_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, mode='a',index_label='index')

        
    