# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:46:43 2022

@author: TOSHIBA-Portégé C30
"""

from network.models import AlexNet, PamapModel, lenet5v, AlexNet_CIFAR10, AlexNet_CIFAR100
import copy


def modelsel(args, device):
    if args.dataset in ['vlcs', 'pacs', 'off_home', 'off-cal', 'covid']:
        server_model = AlexNet(num_classes=args.num_classes).to(device)
    elif 'medmnist' in args.dataset:
        server_model = lenet5v().to(device)
    elif 'pamap' in args.dataset:
        server_model = PamapModel().to(device)
    elif 'femnist' in args.dataset:
        server_model = lenet5v().to(device)
    elif 'cifar10' in args.dataset:
        server_model = AlexNet_CIFAR10().to(device)
    elif 'cifar100' in args.dataset:
        server_model = AlexNet_CIFAR100().to(device)

    client_weights = [1/args.n_clients for _ in range(args.n_clients)]
    models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    return server_model, models, client_weights