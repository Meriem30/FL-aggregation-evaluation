
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import pandas as pd

def plotOneAlg(file_name, metric,marker,label,all_ax ):
    #read the csv file to plot the result for one algo
    dataframe = pd.read_csv(file_name +"/acc.csv")
    nb_rounds = dataframe['n_rounds']
    test_acc = dataframe['avg-test-accuracy']
    nb_epochs = list(set(dataframe['num-epochs']))
    nb_batch = list(set(dataframe['batch-size']))
    time_consumed = dataframe['time-S']
    conv_rate = dataframe['convergence-rate']
    style = [(0, ()) , (0, (1, 1)) , (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),
             (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10)),(0, (3, 1, 1, 1, 1, 1)),(5,(10,3))]

    global bj
    global bk
    global be
    global bb
    if metric == 'conv-rate':
            for epoch in range(len(nb_epochs)):
                if nb_epochs[epoch] == bestE:
                    be = True
                for batch in range(len(nb_batch)):
                    if nb_batch[batch] == bestB:
                        bb = True
                    if be == True & bb == True:
                        bj = ((epoch +batch) * len(set(nb_rounds)))
                        bk = bj + len(set(nb_rounds))
                    # plot each algo in separate figure
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    for i in range(len(nb_epochs)* len(nb_batch)):
                        if i == 0:
                            j= i
                        else:
                            j= k
                        k = (i +1) * len(set(nb_rounds))
                        ax.plot(nb_rounds.iloc[j:k], time_consumed.iloc[j:k], marker=marker,  label=label + ' Max Acc ' + str("{:.2f}".format(max(test_acc[j:k])* 100))+ ' %')#+' E = ' + str(nb_epochs[epoch] )+ ' B = ' +str(nb_batch[batch]))
                        plt.xlabel('Number of Iterations')
                        plt.ylabel('Convergence Rate (time in seconds)')
                        # choose the best values of E (epoch) and B (batch size) to plot all the algo in the same fig
                    plt.legend()
                    fig.savefig(file_name + f'/conv_rate_E_{nb_epochs[epoch]}_B_{nb_batch[batch]}.png', bbox_inches='tight')
            all_ax.plot(nb_rounds.iloc[bj:bk], time_consumed.iloc[bj:bk], marker=marker)  # +' E = ' + str(nb_epochs[epoch] )+ ' B = ' +str(nb_batch[batch]))


def plotResults(files, algos, num_algs):
    #n_rounds = [3, 5, 7, 10]
    marker = ["D", "o", "v", "p", "*", "d", ',', 'P']
    labels = algos
    all_fig = plt.figure()
    all_ax = all_fig.add_subplot(1, 1, 1)

    for i in range(num_algs):
        metric = 'conv-rate'
        colors = mcolors.TABLEAU_COLORS
        plotOneAlg(files[i], metric, marker[i], labels[i], all_ax)
        all_ax.legend(algos)
        all_fig.savefig(files[num_algs - 1] + '/conv_rate_all_algo_E_.png',
                        bbox_inches='tight')




if __name__ == '__main__':
    lines = 0
    bestE = 1
    bestB = 32
    bj = 0
    bk = 0
    be = False
    bb = False
    df = pd.read_csv("results/conv_rate/name_file_res_algos.csv")
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
