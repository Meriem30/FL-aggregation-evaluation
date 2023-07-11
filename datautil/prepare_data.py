# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:27:33 2022

@author: TOSHIBA-Portégé C30
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from datautil.datasplit import getdataloader
from PIL import ImageFile
import  os

from datautil.partitioned_femnist import PartitionedFEMNIST
from datautil.partitioned_CIFAR import PartitionCIFAR

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data(data_name):
    """Return the algorithm class with the given name."""
    datalist = { 'medmnist': 'medmnist','medmnistA': 'medmnist', 'medmnistC': 'medmnist',
                 'pamap': 'pamap', 'femnist' : 'femnist', 'cifar10' : 'cifar', 'cifar100': 'cifar'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Dataset not found: {}".format(data_name))
    return globals()[datalist[data_name]]


def gettransforms():
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test


class mydataset(object):
    def __init__(self, args):
        self.x = None
        self.targets = None
        self.dataset = None
        self.transform = None
        self.target_transform = None
        self.loader = None
        self.args = args

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        x = self.input_trans(self.loader(self.x[index]))
        ctarget = self.target_trans(self.targets[index])
        return x, ctarget

    def __len__(self):
        return len(self.targets)


class ImageDataset(mydataset):
    def __init__(self, args, dataset, root_dir, domain_name):
        super(ImageDataset, self).__init__(args)
        self.imgs = ImageFolder(root_dir+domain_name).imgs
        self.domain_num = 0
        self.dataset = dataset
        imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.targets = np.array(labels)
        transform, _ = gettransforms()
        target_transform = None
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.pathx = imgs
        self.x = self.pathx
        
#adapt from the tlbook-code src code        
def get_data_medmnist(file):
    data= np.load(file)
    print('mnist data type', type(data))
    train_data=np.vstack((data['train_images'],data['val_images'],data['test_images']))
    print('mnist train_data type', type(train_data))

    y=np.hstack((np.squeeze(data['train_labels']),np.squeeze(data['val_labels']),np.squeeze(data['test_labels'])))
    return train_data,y

def get_data_pamap(file):
    x = np.load(file+'x.npy')
    y = np.load(file+'y.npy')
    return x, y


class MedMnistDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.data,self.targets=get_data_medmnist(filename+'organcmnist.npz')
        #self.data = np.load(filename+'xdata.npy')
        #self.targets = np.load(filename+'ydata.npy')
        self.targets = np.squeeze(self.targets)
        self.transform = transform

        self.data = torch.Tensor(self.data)
        self.data = torch.unsqueeze(self.data, dim=1)

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
class Femnist(Dataset):
    def __init__(self, args, transform=None):
            femnist(args)


class Cifar10(Dataset):
    def __init__(self, args, transform=None):
            cifar(args)
class Cifar100(Dataset):
    def __init__(self, args, transform=None):
            cifar(args)
class Cifar(Dataset):
    def __init__(self, args, transform=None):
            cifar(args)



class PamapDataset(Dataset):
    def __init__(self, filename, transform=None):
            self.data, self.targets = get_data_pamap(filename)

            #self.data = np.load(filename+'x.npy')
            #self.targets = np.load(filename+'y.npy')
            self.select_class()
            self.transform = transform
            self.data = torch.Tensor(self.data)
            self.data = torch.unsqueeze(self.data, dim=1)
            #self.data = self.data.permute(0,2,1)

            self.data = torch.einsum('bxyz->bzxy', self.data)


    def select_class(self):
        xiaochuclass = [0, 5, 12]
        index = []
        for ic in xiaochuclass:
            index.append(np.where(self.targets == ic)[0])
        index = np.hstack(index)
        allindex = np.arange(len(self.targets))
        allindex = np.delete(allindex, index)
        self.targets = self.targets[allindex]

        self.data = self.data[allindex]
        ry = np.unique(self.targets)
        ry2 = {}
        for i in range(len(ry)):
            ry2[ry[i]] = i
        for i in range(len(self.targets)):
            self.targets[i] = ry2[self.targets[i]]

    def __len__(self):
        self.filelength = len(self.targets)
        return self.filelength

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def getfeadataloader(args):
    trl, val, tel = [], [], []
    trd, vad, ted = [], [], []
    for item in args.domains:
        data = ImageDataset(args, args.dataset,
                            args.root_dir+args.dataset+'/', item)
        l = len(data)
        index = np.arange(l)
        np.random.seed(args.seed)
        np.random.shuffle(index)
        l1, l2, l3 = int(l*args.datapercent), int(l *
                                                  args.datapercent), int(l*0.2)
        trl.append(torch.utils.data.Subset(data, index[:l1]))
        val.append(torch.utils.data.Subset(data, index[l1:l1+l2]))
        tel.append(torch.utils.data.Subset(data, index[l1+l2:l1+l2+l3]))
        _, target_transform = gettransforms()
        val[-1].transform = target_transform
        tel[-1].transform = target_transform
        trd.append(torch.utils.data.DataLoader(
            trl[-1], batch_size=args.batch, shuffle=True))
        vad.append(torch.utils.data.DataLoader(
            val[-1], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[-1], batch_size=args.batch, shuffle=False))
    return trd, vad, ted


def img_union(args):
    return getfeadataloader(args)


def getlabeldataloader(args, data):
    trl, val, tel = getdataloader(args, data)
    print('train',len(trl[0]))
    print('val',len(val[0]))
    print('test',len(tel[0]))
    trd, vad, ted = [], [], []
    for i in range(len(trl)):
        trd.append(torch.utils.data.DataLoader(
            trl[i], batch_size=args.batch, shuffle=False))
        vad.append(torch.utils.data.DataLoader(
            val[i], batch_size=args.batch, shuffle=False))
        ted.append(torch.utils.data.DataLoader(
            tel[i], batch_size=args.batch, shuffle=False))
    return trd, vad, ted


def medmnist(args):
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 11
    return trd, vad, ted
def pamap(args):
    data = PamapDataset(args.root_dir+args.dataset+'/')
    print('data from pamap function', len(data.targets))
    trd, vad, ted = getlabeldataloader(args, data)
    args.num_classes = 10
    return trd, vad, ted

def femnist(args):
    path = os.path.join("./split/", "femnist_{}_{}_{}".format(args.datapercent, args.n_clients, args.partition_data))
    fed_femnist = PartitionedFEMNIST(root="./data/femnist/",
                                             path=path ,
                                             num_clients=args.n_clients,
                                             partition=args.partition_data , # "unbalance", "noniid-labeldir"
                                             dir_alpha=args.diralpha,
                                             seed=args.seed,
                                             #preprocess=args.preprocess,
                                             download=True,
                                             verbose=False,
                                             transform=transforms.Compose(
                                                 [transforms.ToPILImage(), transforms.ToTensor()]))
    trd, vald, testd = [], [], []
    for idx in range(args.n_clients):
        #dataset_train = fed_femnist.get_dataset(idx, 'train')
        #dataset_val = fed_femnist.get_dataset(idx, 'val')
        #dataset_test = fed_femnist.get_dataset(idx, 'test')
        train_loader = fed_femnist.get_dataloader(idx, args.batch, 'train')
        val_loader = fed_femnist.get_dataloader(idx, args.batch, 'val')
        test_loader = fed_femnist.get_dataloader(idx, args.batch, 'test')
        trd.append(train_loader)
        vald.append(val_loader)
        testd.append(test_loader)
    print("Train set loader size for ith client:", len(trd[0]))
    # print("Validation set  loader size:", len(val_data))
    print("Test set loader size  for ith client:", len(testd[0]))
    return trd, vald, testd

def cifar(args):
    path = os.path.join("./split/", "{}_n_clients{}_balance{}_{}".format(args.dataset,args.n_clients,args.balance,args.partition_data))
    root_path = os.path.join("./data/", "{}".format(args.dataset))
    fed_cifar = PartitionCIFAR(root=root_path,
                                             path=path ,
                                             dataname= args.dataset,
                                             num_clients=args.n_clients,
                                             partition=args.partition_data , # "unbalance", "noniid-labeldir"
                                             dir_alpha=args.diralpha,
                                             unbalance_sgm=args.unbalance_sgm,
                                             num_shards=args.num_shards,
                                             seed=args.seed,
                                             preprocess=args.preprocess,
                                             balance=args.balance,
                                             download=args.download,
                                             verbose=args.verbose,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor()]),
                                             target_transform=None)
    trd, vald, testd = [], [], []
    for idx in range(args.n_clients):
        #dataset_train = fed_cifar.get_dataset(idx, 'train')
        #dataset_val = fed_cifar.get_dataset(idx, 'val')
        #dataset_test = fed_cifar.get_dataset(idx, 'test')
        train_loader = fed_cifar.get_dataloader(idx, args.batch, 'train')
        val_loader = fed_cifar.get_dataloader(idx, args.batch, 'val')
        test_loader = fed_cifar.get_dataloader(idx, args.batch, 'test')
        trd.append(train_loader)
        vald.append(val_loader)
        testd.append(test_loader)
    print("Train set loader size for ith client:", len(trd[0]))
    # print("Validation set  loader size:", len(val_data))
    print("Test set loader size  for ith client:", len(testd[0]))
    return trd, vald, testd



class combinedataset(mydataset):
    def __init__(self, datal, args):
        super(combinedataset, self).__init__(args)

        self.x = np.hstack([np.array(item.x) for item in datal])
        self.targets = np.hstack([item.targets for item in datal])
        s = ''
        for item in datal:
            s += item.dataset+'-'
        s = s[:-1]
        self.dataset = s
        self.transform = datal[0].transform
        self.target_transform = datal[0].target_transform
        self.loader = datal[0].loader


def getwholedataset(args):
    datal = []
    for item in args.domains:
        datal.append(ImageDataset(args, args.dataset,
                     args.root_dir+args.dataset+'/', item))
    # data=torch.utils.data.ConcatDataset(datal)
    data = combinedataset(datal, args)
    return data


def img_union_w(args):
    return getwholedataset(args)


def medmnist_w(args):
    data = MedMnistDataset(args.root_dir+args.dataset+'/')
    args.num_classes = 11
    return data



def pamap_w(args):
    data = PamapDataset(args.root_dir+args.dataset+'/')
    args.num_classes = 10
    return data

def get_whole_dataset(data_name):
    datalist = { 'medmnist': 'medmnist_w','medmnistA': 'medmnist_w','medmnistC': 'medmnist_w'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(data_name))
    return globals()[datalist[data_name]]