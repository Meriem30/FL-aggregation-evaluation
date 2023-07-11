# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:44:03 2022

@author: TOSHIBA-Portégé C30
"""


import torch
import copy

def communication(args, server_model, models, client_weights):
    client_num=len(models)
    with torch.no_grad():
        if args.alg.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower()=='fedap':
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        if 'bn' not in key:
                            models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower()=='powerofchoice':
            client_num = len(models)
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        if client_idx in args.list_selected_clients:
                            temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                            server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        if client_idx in args.list_selected_clients:
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower()=='feddyn':

            server_model_tensor = torch.cat([param.data.flatten() for param in server_model.parameters()])
            temp = torch.zeros_like(server_model_tensor)
            local_model_tensors = []
            for model in models:
                flattened_params = torch.cat([param.data.flatten() for param in model.parameters()])
                local_model_tensors.append(flattened_params)
            args.h = args.h - args.alpha * (1.0 / args.n_clients) * (
                sum([local_model_tensors[i] - server_model_tensor for i in range(len(local_model_tensors))]))
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * local_model_tensors[client_idx]
            temp = temp - 1.0 / args.alpha * args.h
            idx = 0
            for param in server_model.parameters():
                param.data.copy_(temp[idx:idx + param.data.numel()].reshape(param.data.shape))
                idx += param.data.numel()
            for client_idx in range(len(client_weights)):
                idx = 0
                for param in models[client_idx].parameters():
                    param.data.copy_(temp[idx:idx + param.data.numel()].reshape(param.data.shape))
                    idx += param.data.numel()

        else:
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models
    