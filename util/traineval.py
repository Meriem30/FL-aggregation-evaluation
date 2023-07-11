# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:58:49 2022

@author: TOSHIBA-Portégé C30
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datautil.datasplit import define_pretrain_dataset
from datautil.prepare_data import get_whole_dataset


def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        if len(data_loader) == 0:
            return 0.0, 0.0
        else:
            return loss_all / len(data_loader), correct/total

def train_dyn(args, model, server_model, data_loader, optimizer, loss_fun, device, L, c_idx):
    # L has to be initiated in the init functionof the alg feddyn
    L_t = None
    if L[c_idx] is None:
        parameters_tensor = torch.cat([param.data.flatten() for param in server_model.parameters()])
        L[c_idx] = torch.zeros_like(parameters_tensor)

    L_t = L[c_idx]
    # no need to save the current local  model cause here we have the parameter of server model, so
    # it is the same parameter as the previous local model we are freezing here before a new iteration
    #frz_parameters = model
    server_model_tensor = torch.cat([param.data.flatten() for param in server_model.parameters()])
    local_model_tensor = torch.cat([param.data.flatten() for param in model.parameters()])
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        #min_value = min(target)
        #max_value = max(target)
        #print("Min value for target:", min_value)
        #print("Max value for target:", max_value)

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        #print('output type ', type(output))
        #min_value = torch.min(output).item()
        #max_value = torch.min(output).item()
        #print("Min value for output:", min_value)
        #print("Max value for output:", max_value)
        l1 = loss_fun(output, target)
        l2 = torch.dot(L_t, local_model_tensor)
        #l3 = torch.sum(torch.pow(model- frz_parameters, 2))
        l3 = torch.sum(torch.pow(local_model_tensor - server_model_tensor, 2))

        loss = l1 - l2 + 0.5 * args.alpha * l3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #L[c_idx] = L_t - args.alpha * (model- frz_parameters)
        L[c_idx] = L_t - args.alpha * (local_model_tensor - server_model_tensor)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    if len(data_loader) == 0:
        return 0.0, 0.0
    else:
        return loss_all / len(data_loader), correct / total
def train_prox(args, model, server_model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total


def trainwithteacher(model, data_loader, optimizer, loss_fun, device, tmodel, lam, args, flag):
    model.train()
    if tmodel:
        tmodel.eval()
        if not flag:
            with torch.no_grad():
                for key in tmodel.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    elif args.nosharebn and 'bn' in key:
                        pass
                    else:
                        model.state_dict()[key].data.copy_(
                            tmodel.state_dict()[key])
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        f1 = model.get_sel_fea(data, args.plan)
        loss = loss_fun(output, target)
        if flag and tmodel:
            f2 = tmodel.get_sel_fea(data, args.plan).detach()
            loss += (lam*F.mse_loss(f1, f2))
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def pretrain_model(args, model, filename, device='cuda'):
    print('===training pretrained model===')
    data = get_whole_dataset(args.dataset)(args)
    predata = define_pretrain_dataset(args, data)
    traindata = torch.utils.data.DataLoader(
        predata, batch_size=args.batch, shuffle=True)
    loss_fun = nn.CrossEntropyLoss()
    opt = optim.SGD(params=model.parameters(), lr=args.lr)
    for _ in range(args.pretrained_iters):
        _, acc = train(model, traindata, opt, loss_fun, device)
    torch.save({
        'state': model.state_dict(),
        'acc': acc
    }, filename)
    print('===done!===')


