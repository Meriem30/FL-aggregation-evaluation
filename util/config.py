# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:29:33 2022

@author: TOSHIBA-Portégé C30
"""


import random
import numpy as np
import torch


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'dg4':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn']
  
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'dg4': ['mnist', 'mnist_m', 'svhn', 'syn']
    }
    if dataset in ['dg5', 'dg4']:
        args.shuffle_shape = (3, 36, 36)
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    return args


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False