# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 08:11:16 2022

@author: TOSHIBA-Portégé C30
"""
from matplotlib import pyplot as plt

import pandas as pd

def plotOneAlg(file_name, metric,marker,label,ax ):
    #read the csv file to plot the result for one algo
    dataframe = pd.read_csv(file_name +"/acc.csv")
    test_acc = dataframe['avg-test-accuracy']
    nb_rounds = dataframe['n_rounds']

    train_loss = dataframe['avg-train-loss']
    if metric == 'acc':
        ax.plot(nb_rounds, test_acc, marker=marker, label=label)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Average Test Accuracy')
    elif metric == 'loss':
        ax.plot(nb_rounds, train_loss, marker=marker, label=label)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
    plt.legend()

def plotResults(files, algos, num_algs):
    #n_rounds = [3, 5, 7, 10]
    marker = ["D", "o", "v", "p", "*", "d", ',', 'P']
    labels = algos
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #for each algo(one line) in the results-file-name plot the acc results in one figure
    for i in range(num_algs):
        metric = 'acc'
        plotOneAlg(files[i],metric,marker[i],labels[i],ax)
        fig.savefig(files[num_algs-1] + '/n_rounds_avg_accu.png', bbox_inches='tight')
    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    for i in range(num_algs):
        metric = 'loss'
        plotOneAlg(files[i],metric,marker[i],labels[i],ax2)
        fig.savefig(files[num_algs-1] + '/n_epochs_avg_loss.png', bbox_inches='tight')

if __name__ == '__main__':
    lines = 0

    df = pd.read_csv("results/n_rounds/name_file_res_algos.csv")
    # count lines (= to number of algos) in the results-file-name
    lines = len(df)
    algos = df['algo-name'].values.tolist()
    # print(algos)
    files = df['results-file-name'].values.tolist()
    # print(files)
    plotResults(files, algos, lines)


    #plot if results-file-name is a .txt file
    #with open("results/n_clients/name_file_res_algos.txt", mode='r') as f:
    #    files = [x.strip() for x in f.readlines()]
    #    for x in files:
    #        if x != 'f.txt':
    #            lines += 1
    #f.close()
    #print(files)
    #plotResults(files,lines)
