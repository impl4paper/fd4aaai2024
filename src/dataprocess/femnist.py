from torchvision.datasets import MNIST, utils
from PIL import Image
import os.path
import torch
import scipy.io as sio
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import datasets, transforms







def cifar_noniid_fix(d, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 20, 150
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    
    data_y =[]
    for i in range(len(d)):
        x, y  = d[i]
        data_y.append(y)
    labels = np.array(data_y)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users



class DatasetSplit_fix(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(int(label))#  根据imbalanced-cifar10修改了这里

def numpy2dataset(N_parties=10, cls_use =[i for i in range(10,16)], num_samples=20):
    train_x, train_y, test_x, test_y, w_t, w_ = load_EMNIST_data()
    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    p,pv = generate_EMNIST_writer_based_data(train_x, train_y, w_t, N_parties=N_parties, classes_in_use=cls_use, N_priv_data_min=num_samples)
    data_x, data_y = generate_partial_data(test_x, test_y, class_in_use=cls_use)
    trainset, testset = [], []
    #x, y = [], []
    num_dict = defaultdict(list)
    start = 0
    end = 0
    for p_i in range(len(p)):
        for i in range(len(p[p_i]['X'])):
            trainset.append((p[p_i]['X'][i], p[p_i]['y'][i]))
        start = end
        end += len(p[p_i]['X'])
        num_dict[p_i] = [j for j in range(start, end)]
    
    trainset = np.array(trainset)
    for i in range(len(data_x)):
        testset.append((data_x[i], data_y[i]))
    testset = np.array(testset)
    testset = DatasetSplit_fix(testset, [i for i in range(len(testset))])
    return trainset, testset, num_dict
    #x = np.array(x)
    #y = np.array(y)
    #dataset['image'] = x
    #dataset['label'] = y

    #testdataset['image'] = data_x
    #testdataset['label'] = data_y
    #return dataset, testdataset, num_dict

    

def noiddcifar100(N_parties=10, cls_use=[0,1, 7, 9, 12,18], num_samples=20):
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

    X_train, y_train= trainset.data,trainset.targets
    X_test, y_test = testset.data,testset.targets
    x_t = np.transpose(X_train, (0,3,1,2))
    y_t = np.array(y_train)
    x_v = np.transpose(X_test, (0,3,1,2))
    y_v = np.array(y_test)
    x_t_p, y_t_p = generate_partial_data(X=x_t, y=y_t, class_in_use=cls_use, verbose=True)
    x_v_p , y_v_p = generate_partial_data(X=x_v, y=y_v, class_in_use=cls_use, verbose=True)
    private_cls_len = len(cls_use)
    public_cls_len = 10
    for index, cls_ in enumerate(cls_use):
        y_t_p[y_t_p == cls_] = index + public_cls_len
        y_t_p[y_t_p == cls_] = index + public_cls_len
    del index, cls_

    mod_private_classes = np.arange(private_cls_len) + public_cls_len
    users_index = cifar_noniid_fix(y_t_p, N_parties)

    private_data, total_private_data  = get_sample_data(x_t_p, y_t_p, users_index, num_samples*18)
    trainset = []
    for p_i in range(len(private_data)):
        for i in range(len(private_data[p_i]['X'])):
            trainset.append((private_data[p_i]['X'][i], private_data[p_i]['y'][i]))
    trainset = np.array(trainset)

    testset = []
    for i in range(len(x_v_p)):
        testset.append((x_v_p[i], y_v_p[i]))
    testset = np.array(testset)
    testset = DatasetSplit_fix(testset, [i for i in range(len(testset))])
    return trainset, testset, users_index
    






    

def get_sample_data(X_train_private, y_train_private,users_index,N_samples_per_class=10):
    private_data,total_private_data = [],{'X':[],'y':[],'idx':[]}
    for k in users_index.keys():
        index = users_index[k]
        idx = np.random.choice(range(len(index)),N_samples_per_class)
        index_o = index[idx].astype(int)
        private_data.append({'X':X_train_private[index_o],'y':y_train_private[index_o],'idx':index_o})
        total_private_data['X'].extend(X_train_private[index_o].tolist())
        total_private_data['y'].extend(y_train_private[index_o].tolist())
        total_private_data['idx'].extend(index_o.tolist())

    return private_data, total_private_data












class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        # if not self._check_exists():
        #     raise RuntimeError('Dataset not found.' +
        #                        ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        #utils.makedir_exist_ok(self.raw_folder)
        #utils.makedir_exist_ok(self.processed_folder)
        if not os.path.exists(self.raw_folder): 
            os.makedirs(self.raw_folder)
        if not os.path.exists(self.processed_folder): 
            os.makedirs(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)



def load_EMNIST_data(file=r'./data/emnist-letters.mat', verbose = True, standarized = False):
    """
    file should be the downloaded EMNIST file in .mat format.
    """    
    mat = sio.loadmat(file)
    data = mat["dataset"]
    

    writer_ids_train = data['train'][0,0]['writers'][0,0]
    writer_ids_train = np.squeeze(writer_ids_train)
    X_train = data['train'][0,0]['images'][0,0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order = "F")
    X_train = X_train[:, np.newaxis, :, :]
    y_train = data['train'][0,0]['labels'][0,0]
    y_train = np.squeeze(y_train)
    y_train -= 1 #y_train is zero-based
    
    writer_ids_test = data['test'][0,0]['writers'][0,0]
    writer_ids_test = np.squeeze(writer_ids_test)
    X_test = data['test'][0,0]['images'][0,0]
    X_test= X_test.reshape((X_test.shape[0], 28, 28), order = "F")
    X_test = X_test[:, np.newaxis, :, :]
    y_test = data['test'][0,0]['labels'][0,0]
    y_test = np.squeeze(y_test)
    y_test -= 1 #y_test is zero-based

    
    if standarized: 
        X_train = X_train/255
        X_test = X_test/255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image
    

    if verbose == True: 
        print("EMNIST-letter dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)
    
    return X_train, y_train, X_test, y_test, writer_ids_train, writer_ids_test




def generate_EMNIST_writer_based_data(X, y, writer_info, N_priv_data_min = 30, 
                                      N_parties = 5, classes_in_use = range(6)):
    
    # mask is a boolean array of the same shape as y
    # mask[i] = True if y[i] in classes_in_use
    mask = None
    mask = [y == i for i in classes_in_use]
    mask = np.any(mask, axis = 0)
    
    df_tmp = None
    df_tmp = pd.DataFrame({"writer_ids": writer_info, "is_in_use": mask})
    #print(df_tmp.head())
    groupped = df_tmp[df_tmp["is_in_use"]].groupby("writer_ids")
    
    # organize the input the data (X,y) by writer_ids.
    # That is, 
    # data_by_writer is a dictionary where the keys are writer_ids,
    # and the contents are the correcponding data. 
    # Notice that only data with labels in class_in_use are included.
    data_by_writer = {}
    writer_ids = []
    for wt_id, idx in groupped.groups.items():
        if len(idx) >= N_priv_data_min:  
            writer_ids.append(wt_id)
            data_by_writer[wt_id] = {"X": X[idx], "y": y[idx], 
                                     "idx": idx, "writer_id": wt_id}
            
    # each participant in the collaborative group is assigned data 
    # from a single writer.
    ids_to_use = np.random.choice(writer_ids, size = N_parties, replace = False)
    combined_idx = np.array([], dtype = np.int64)
    private_data = []
    for i in range(N_parties):
        id_tmp = ids_to_use[i]
        private_data.append(data_by_writer[id_tmp])
        combined_idx = np.r_[combined_idx, data_by_writer[id_tmp]["idx"]]
        del id_tmp
    
    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return private_data, total_priv_data






def generate_partial_data(X, y, class_in_use = None, verbose = False):
    if class_in_use is None:
        idx = np.ones_like(y, dtype = bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis = 0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose == True:
        print("X shape :", X_incomplete.shape)
        print("y shape :", y_incomplete.shape)
    return X_incomplete, y_incomplete
