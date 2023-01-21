# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 08:11:16 2022

@author: TOSHIBA-Portégé C30
"""
from matplotlib import pyplot as plt

import pandas

def plotOneAlg(file_name, nb_clients, marker,label,ax ):
    df = pandas.read_csv(file_name +"/acc.csv")
    test_acc = df['avg-test-accuracy']
    ax.plot(nb_clients, test_acc, marker=marker, label=label)
    plt.legend()

def plotResults(files, num_algs):
    nb_clients = [3, 5, 7, 10]
    marker = ["D", "o", "v", "p", "*", "d"]
    labels = ['FedAvg','FedProx','Fedbn', 'Fedap','MetaFed']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(num_algs):
        plotOneAlg(files[i],nb_clients,marker[i],labels[i],ax)
        plt.xlabel('Number Of Clients')
        plt.ylabel('Test Accuracy')
        fig.savefig(files[num_algs-1] + '/n_clients_avg_accu.png', bbox_inches='tight')


if __name__ == '__main__':
    lines = 0
    with open("results/n_clients/name_file_res_algos.txt", mode='r') as f:
        files = [x.strip() for x in f.readlines()]
        for x in files:
            if x != 'f.txt':
                lines += 1
    f.close()
    print(files)
    plotResults(files, lines)