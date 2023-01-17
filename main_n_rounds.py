# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:26:04 2022

@author: TOSHIBA-Portégé C30
"""

import os
import numpy as np
import torch
import argparse
import pandas as pd
import csv
import datetime

from datautil.prepare_data import *
from util.config import img_param_init, set_random_seed
from util.evalandprint import evalandprint
from alg import algs

from n_clients_plot_acc import plotResults

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed ]')
    parser.add_argument('--datapercent', type=float,
                        default=1, help='data percent to use')
    parser.add_argument('--dataset', type=str, default='medmnist',
                        help='[ medmnist ]')
    parser.add_argument('--root_dir', type=str,
                        default='./data/', help='data path')
    parser.add_argument('--save_path', type=str,
                        default='./cks/', help='path to save the checkpoint')
    parser.add_argument('--device', type=str,
                        default='cpu', help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--n_clients', type=int,
                        default=10, help='number of clients')
    parser.add_argument('--dropout_clients', type=int,
                        default=1, help='client dropout percentage')
    parser.add_argument('--non_iid_alpha', type=float,
                        default=0.1, help='data split for label shift')
    parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way')
    parser.add_argument('--plan', type=int,
                        default=10, help='choose the feature type')
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')

    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='init lam, hyperparmeter for metafed')
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
    args = parser.parse_args()

    args.n_clients = args.n_clients * args.dropout_clients

    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    #

    exp_folder = f'fed_{args.dataset}_{args.alg}_{args.datapercent}_{args.non_iid_alpha}_{args.mu}_{args.model_momentum}_{args.plan}_{args.lam}_{args.threshold}_{args.iters}_{args.wk_iters}'
    if args.nosharebn:
        exp_folder += '_nosharebn'
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, args.alg)

    train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)

    algclass = algs.get_algorithm_class(args.alg)(args)

    if args.alg == 'fedap':
        algclass.set_client_weight(train_loaders)
    elif args.alg == 'metafed':
        algclass.init_model_flag(train_loaders, val_loaders)
        args.iters = args.iters - 1
        print('Common knowledge accumulation stage')

    #
    test_acc = [0] * 10
    start_tuning = 0
    #n_rounds = [100,200,300,400,500]
    n_rounds = [3,5,7,10,15]
    mean_acc_test = [0] * len(n_rounds)
    for i in range(start_tuning, len(n_rounds)):

        best_changed = False

        args.iters = n_rounds[i]
        best_acc = [0] * args.iters
        best_tacc = [0] * args.iters
        mean_acc_test = 0

        start_iter = 0
        print("here in l1 before l1")
        for a_iter in range(start_iter, args.iters):
            print(f"============ Train round {a_iter} ============")
            print('here in l1 and l2')
            print('n_client : ', args.n_clients)
            if args.alg == 'metafed':
                for c_idx in range(args.n_clients):
                    print('c_idx : ', c_idx)
                    algclass.client_train(
                        c_idx, train_loaders[algclass.csort[c_idx]], a_iter)
                algclass.update_flag(val_loaders)
            else:
                # local client training
                for wi in range(args.wk_iters):
                    for client_idx in range(args.n_clients):
                        print('client_idx : ', client_idx)
                        algclass.client_train(
                            client_idx, train_loaders[client_idx], a_iter)

                # server aggregation
                algclass.server_aggre()

            best_acc, best_tacc, best_changed = evalandprint(
                args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
                best_changed)

        if args.alg == 'metafed':
            print('Personalization stage')
            for c_idx in range(args.n_clients):
                algclass.personalization(
                    c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
            best_acc, best_tacc, best_changed = evalandprint(
                args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
                best_changed)

        s = 'Personalized test acc for each client: '
        for item in best_tacc:
            s += f'{item:.4f},'
        mean_acc_test = np.mean(np.array(best_tacc))
        s += f'\nAverage accuracy: {mean_acc_test:.4f}'
        print(s)

        print('my results: ', mean_acc_test)
        test_acc[i] = mean_acc_test

    test_acc = [0] * args.iters


    # store results
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(os.path.dirname(__file__),
                                  "results/n_rounds/accuracy_results_{args.alg}_{args.datapercent}_{args.non_iid_alpha}_{args.mu}_{args.iters}_{args.wk_iters}.csv" + str(
                                      date))
    os.mkdir(results_folder)

    # name file to record reslts
    file_fedavg, file_fedprox, file_fedbn, file_fedap, file_metafed = ''

    if args.alg.equals('fedavg'):
        file_fedavg = results_folder
    elif args.alg.equals('fedprox'):
        file_fedprox = results_folder
    elif args.alg.equals('fedbn'):
        file_fedbn = results_folder
    elif args.alg.equals('fedap'):
        file_fedap = results_folder
    elif args.alg.equals('metafed'):
        file_metafed = results_folder

    with open(results_folder, mode='w') as f:
        fieldnames = ['num_rounds', 'avg-test-accuracy']
        csv_writer = csv.DictWriter(f, fieldnames)

        csv_writer.writeheader()

        csv_writer.writerow(
            {'num_rounds': args.iters, 'avg-test-accuracy': test_acc[0]})
        csv_writer.writerow(
            {'num_rounds': args.iters, 'avg-test-accuracy': test_acc[1]})
        csv_writer.writerow(
            {'num_rounds': args.iters, 'avg-test-accuracy': test_acc[2]})
        csv_writer.writerow(
            {'num_rounds': args.iters, 'avg-test-accuracy': test_acc[3]})

    f.close()

    if args.alg.equals('metafed'):
        plotResults(file_fedavg, file_fedprox, file_fedbn, file_fedap, file_metafed)

# run : python main_n_rounds.py --alg fedavg --dataset medmnist  --wk_iters 1 --non_iid_alpha 0.1