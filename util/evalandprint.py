# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:30:03 2022

@author: TOSHIBA-Portégé C30
"""

import enum
import numpy as np
import torch


def evalandprint(args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter, best_changed,
                 train_loss, val_loss):
    # evaluation on training data
    train_loss_list = [None] * args.n_clients
    for client_idx in range(args.n_clients):
        train_loss, train_acc = algclass.client_eval(
            client_idx, train_loaders[client_idx])
        train_loss_list[client_idx] = train_loss
        print(' Site-{:02d} | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(client_idx,train_loss,train_acc))
    train_loss = train_loss_list


    # evaluation on valid data
    val_acc_list = [None] * args.n_clients
    val_loss_list = [None] * args.n_clients

    for client_idx in range(args.n_clients):
        val_loss, val_acc = algclass.client_eval(
            client_idx, val_loaders[client_idx])
        val_acc_list[client_idx] = val_acc
        val_loss_list[client_idx] = val_loss
        print(' Site-{:02d} | Val Loss: {:.4f} | Val Acc: {:.4f}'.format(client_idx,val_loss, val_acc))
    val_loss = val_loss_list

    if np.mean(val_acc_list) > np.mean(best_acc):
        print('len after mean ', val_acc_list)
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = val_acc_list[client_idx]
            best_epoch = a_iter
        best_changed = True 

    if best_changed:
        best_changed = False
        # test
        for client_idx in range(args.n_clients):
            _, test_acc = algclass.client_eval(
                client_idx, test_loaders[client_idx])
            print('Test site-{:02d} | Epoch:{} | Test Acc: {:.4f}'.format(client_idx,best_epoch, test_acc))
            best_tacc[client_idx] = test_acc


        print('Saving the local and server checkpoint to {}'.format(SAVE_PATH))
        tosave = {'best_epoch': best_epoch, 'best_acc': best_acc, 'best_tacc': np.mean(np.array(best_tacc))}
        #for i,tmodel in enumerate(algclass.client_model):
         #   tosave['client_model_'+str(i)]=tmodel.state_dict()
            #print(tmodel.state_dict())
        #tosave['server_model']=algclass.server_model.state_dict()
        # Print server model's state_dict
        print("Model's state_dict:")
        for param_tensor in algclass.server_model.state_dict():
            print(param_tensor, "\t", algclass.server_model.state_dict()[param_tensor].size())
            
        torch.save(tosave, SAVE_PATH)


    return best_acc, best_tacc, best_changed, train_loss, val_loss