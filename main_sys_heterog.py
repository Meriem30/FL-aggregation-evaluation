

import os
import numpy as np
import torch
import argparse
import pandas as pd
import csv
import datetime
import math
import random
import time

from datautil.prepare_data import *
from util.config import img_param_init, set_random_seed
from util.evalandprint import evalandprint
from alg import algs

from n_clients_plot_acc import plotResults

if __name__ == '__main__':
    t0 = time.time()
    print('the time now is : ')
    print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed | powerofchoice | feddyn]')
    parser.add_argument('--datapercent', type=float,
                        default=1, help='data percent to use')
    parser.add_argument('--dataset', type=str, default='medmnist',
                        help='[ medmnist , pamap, femnist, cifar10, cifar100]')
    parser.add_argument('--root_dir', type=str,
                        default='./data/', help='data path')
    parser.add_argument('--save_path', type=str,
                        default='./cks/', help='path to save the checkpoint')
    parser.add_argument('--device', type=str,
                        default='cuda', help='[cuda | cpu]')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=100,
                        help='iterations for communication')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--n_clients', type=int,
                        default=10, help='number of clients')
    parser.add_argument('--dropout_clients', type=int,
                        default=0, help='client dropout percentage')
    parser.add_argument('--non_iid_alpha', type=float,
                        default=0.3, help='data split for label shift')
    parser.add_argument('--partition_data', type=str,
                        default='non_iid_dirichlet', help='partition data way')  # 'unbalanced'
    parser.add_argument('--plan', type=int,
                        default=10, help='choose the feature type')
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--nosharebn', action='store_true',
                        help='not share bn')
    parser.add_argument('--diralpha', type=float,
                        default=0.3, help='parameter for Dirichlet distribution')
    parser.add_argument('--preprocess', type=bool,
                        default=False, help='parameter dataset preprocess')
    parser.add_argument('--download', type=bool,
                        default=False, help='parameter for download dataset ')
    parser.add_argument('--num_shards', type=int,
                        default=None,
                        help=' Number of shards in non-iid ``"shards"`` partition. Only works if ``partition=shards')
    parser.add_argument('--verbose', type=bool,
                        default=False,
                        help='Whether to print partition process')
    parser.add_argument('--min_require_size', type=int,
                        default=None,
                        help='Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``. Only works if ``partition="noniid-labeldir"``')
    # unbalance_sgm = 0.3
    parser.add_argument('--unbalance_sgm', type=float,
                        default=0.3,
                        help='Log-normal distribution variance for unbalanced data partition over clients. Default as ``0`` for balanced partition.')
    parser.add_argument('--balance', type=bool,
                        default=False,
                        help='Balanced partition over all clients or not')
    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='init lam, hyperparmeter for metafed')
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
    parser.add_argument('--d', type=int,
                        default=35, help='number of clients to be selected for powerofchoice')

    parser.add_argument('--alpha', type=float,
                        default=1e-2, help='regularization parameter for feddyn')
    parser.add_argument('--major_classes_num', type=int,
                        default=2, help='maximum number of classes in each client for Quantitybased distrinution skew')
    parser.add_argument('--het_level', type=float,
                        default=0, help='level of system heterogeneity')
    # parse to extract arguments

    args = parser.parse_args()

    # get the true number of clients considering the dropout percentage
    _, args.n_clients = math.modf(args.n_clients * (1 - args.dropout_clients))
    args.n_clients = int(args.n_clients)

    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)

    # create folder to save checkpoints
    exp_folder = f'fed_{args.dataset}_{args.alg}_{args.datapercent}_{args.non_iid_alpha}_{args.mu}_{args.model_momentum}_{args.plan}_{args.lam}_{args.threshold}_{args.iters}_{args.epochs}'
    if args.nosharebn:
        exp_folder += '_nosharebn'
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, args.alg)

    # get the prepared dataset
    train_loaders, val_loaders, test_loaders = get_data(args.dataset)(args)
    print('train_loaders', len(train_loaders[0]))
    print('val_loaders', len(val_loaders[0]))
    print('test_loaders', len(test_loaders[0]))

    # get the class of the specified alg
    algclass = algs.get_algorithm_class(args.alg)(args)

    # special params
    if args.alg == 'fedap':
        algclass.set_client_weight(train_loaders)
    elif args.alg == 'metafed':
        algclass.init_model_flag(train_loaders, val_loaders)
        args.iters = args.iters - 1
        print('Common knowledge accumulation stage')
    elif args.alg == 'powerofchoice':
        args.list_selected_clients = list(range(args.n_clients))

    # store results
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    exp_folder = f"accuracy_results_{args.alg}_{args.datapercent}_{args.non_iid_alpha}_{args.mu}_{args.iters}_{args.epochs}" + str(
        date)

    # create folder to save n_clients results
    results_folder = os.path.join(os.path.dirname(__file__), "results","sys_heterog" ,  f"{args.dataset}_balance_{args.balance}_dist_{args.partition_data}_" + exp_folder)
    os.mkdir(results_folder)

    # create a csv (or .txt) file to save the results-file-name for each alg
    res_files_name = "results/sys_heterog/name_file_res_algos.csv"
    # res_files_name = "results/n_clients/name_file_res_algos.txt"

    # write the file_name into a csv file
    with open(res_files_name, newline='', encoding='utf-8', mode='a+') as fn:
        res_fields_name = ['algo-name', 'results-file-name']
        csv_writer = csv.DictWriter(fn, res_fields_name)
        if args.alg == 'fedavg':
            csv_writer.writeheader()
        csv_writer.writerow({'algo-name': args.alg, 'results-file-name': results_folder})
    fn.close()

    # write the file_name into a txt file
    # with open(res_files_name, mode='w') as fn:
    #    fn.write(results_folder)
    # fn.close()

    # create a csv file to save accuracy result for each value of the parameter (n_clients)
    with open(results_folder + "/acc.csv", newline='', encoding='utf-8', mode='w') as f:
        fieldnames = ['n-level', 'origin-epoch', 'rand-epoch','avg-test-accuracy', 'avg-train-loss', 'fairness-var', ]
        csv_writer = csv.DictWriter(f, fieldnames)
        csv_writer.writeheader()

    start_tuning = 0

    n_het_level = [0.1,0.2, 0.3,0.4,0.5, 0.9]
    n_epochs = [5]

    test_acc = [0] * args.n_clients

    for i in range(start_tuning, len(n_het_level)):


            best_changed = False

            args.het_level = n_het_level[i]
            best_acc = [0] * args.n_clients
            best_tacc = [0] * args.n_clients
            mean_acc_test = 0

            train_loss = [0] * args.n_clients
            val_loss = [0] * args.n_clients
            mean_train_loss = 0

            start_iter = 0
            for a_iter in range(start_iter, args.iters):
                print(f"============ Train round {a_iter} ============")

                print('n_clients: ', args.n_clients)
                print('the algo in execution : ', args.alg)
                print(f'System Heterogeneity : epochs {args.epochs }; heterogeneity level {args.het_level}')
                x = random.randint(1, args.epochs)
                print(x)
                normal_wl = int(args.n_clients * (1 - args.het_level))
                partial_wk = args.n_clients - normal_wl
                if args.alg == 'metafed':
                    # local client training for normal worker
                    for epochs in range(args.epochs):
                        for client_idx in range(normal_wl):
                            algclass.client_train(
                                client_idx, train_loaders[algclass.csort[client_idx]], a_iter)
                    # local training for worker with system constraint
                    for epochs in range(partial_wk):
                        args.epochs = x
                        for client_idx in range(normal_wl, args.n_clients):
                            algclass.client_train(
                                client_idx, train_loaders[algclass.csort[client_idx]], a_iter)

                    algclass.update_flag(val_loaders)
                elif args.alg == 'powerofchoice':
                    # local client training
                    list_index_clients = []
                    if a_iter == 0:
                        list_index_clients = algclass.sample_condidates(args)
                        args.list_selected_clients = list_index_clients
                        print("list_index_clients ", args.list_selected_clients)
                        print('number of client to be selected ', args.d)
                    else:
                        condidates = list(range(args.n_clients))
                        list_index_clients = algclass.sample_clients(args, condidates, train_loss)
                        args.list_selected_clients = list_index_clients
                        print("list_index_clients", args.list_selected_clients)
                        print('number of client to be selected ', args.d)
                    # local client training for normal worker
                    for epochs in range(args.epochs):
                        for client_idx in range(normal_wl):
                            if client_idx in list_index_clients:
                                algclass.client_train(
                                    client_idx, train_loaders[client_idx], a_iter)
                            else:
                                pass
                    #local training for worker with system constraint
                    for epochs in range(args.epochs):
                        args.epochs = x
                        for client_idx in range(partial_wk, args.n_clients):
                            if client_idx in list_index_clients:
                                algclass.client_train(
                                    client_idx, train_loaders[client_idx], a_iter)
                            else:
                                pass

                    # server aggregation
                    algclass.server_aggre()


                else:

                    # local client training for normal worker
                    for epochs in range(args.epochs):
                        for client_idx in range(normal_wl):
                            algclass.client_train(
                                client_idx, train_loaders[client_idx], a_iter)
                    #local training for worker with system constraint
                    for epochs in range(partial_wk):
                        args.epochs = x
                        for client_idx in range(normal_wl, args.n_clients):
                            algclass.client_train(
                                client_idx, train_loaders[client_idx], a_iter)

                    # server aggregation
                    algclass.server_aggre()

                best_acc, best_tacc, best_changed, train_loss, val_loss = evalandprint(
                    args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
                    best_changed, train_loss, val_loss)
            if args.alg == 'metafed':
                print('Personalization stage')
                for c_idx in range(args.n_clients):
                    algclass.personalization(
                        c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
                best_acc, best_tacc, best_changed, train_loss, val_loss = evalandprint(
                    args, algclass, train_loaders, val_loaders, test_loaders, SAVE_PATH, best_acc, best_tacc, a_iter,
                    best_changed, train_loss, val_loss)

            s = 'Personalized test acc for each client: '
            for item in best_tacc:
                s += f'{item:.4f},'
            mean_acc_test = np.mean(np.array(best_tacc))
            mean_train_loss = np.mean(np.array(train_loss))
            # variance of the chosen metric (acc) = fairness of the model
            fair_var = np.var(np.array(best_tacc)) * 10000

            s += f'\nAverage accuracy: {mean_acc_test:.4f}'
            print(s)

            print('my results: ', mean_acc_test)

            print(' the average of train loss over all clients :', mean_train_loss)

            print(' the average of accuracy variance ==> fairness :', fair_var)
            # save the accuracy and loss results
            with open(results_folder + "/acc.csv", newline='', encoding='utf-8', mode='a') as f:
                csv_writer = csv.DictWriter(f, fieldnames)
                csv_writer.writerow({'n-level': args.het_level,
                                     'origin-epoch' : n_epochs[0],
                                     'rand-epoch' : x ,
                                     'avg-test-accuracy': mean_acc_test,
                                     'avg-train-loss': mean_train_loss,
                                     'fairness-var': fair_var
                                     })

    # close the results file when the loop is over
    f.close()
    tf = (time.time() - t0) / 60
    print('the time now is : ')
    print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print('the execution takes (min.) ', tf)




