import torch
import torch.nn as nn
import torch.optim as optim

from util.modelsel import modelsel
from util.traineval import train_dyn, test
from alg.core.comm import communication

from alg.fedavg import fedavg

class feddyn(fedavg):
    def __init__(self, args):
        super(feddyn, self).__init__(args)
        self.server_model, self.client_model, self.client_weight = modelsel(
            args, args.device)
        self.optimizers = [optim.SGD(params=self.client_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        # we add two params
        self.L = [None for _ in range(args.n_clients)]
        parameters_tensor = torch.cat([param.data.flatten() for param in self.server_model.parameters()])
        args.h = torch.zeros_like(parameters_tensor)
        print('len parameter_tensor h' , len(args.h))
        self.args = args

    def client_train(self, c_idx, dataloader, round):
        train_loss, train_acc = train_dyn(
            self.args, self.client_model[c_idx], self.server_model, dataloader, self.optimizers[c_idx], self.loss_fun,
            self.args.device, self.L, c_idx)
        return train_loss, train_acc

