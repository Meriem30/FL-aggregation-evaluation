
from matplotlib import pyplot as plt

import pandas as pd

def plotOneAlg(file_name, metric, marker,label,ax, fig):
    #read the csv file to plot the result for one algo
    dataframe = pd.read_csv(file_name +"/acc.csv")
    print(dataframe.columns.tolist())
    n_epochs = list(set(dataframe['origin-epoch']))
    n_het_level = dataframe['n-level']
    test_acc = dataframe['avg-test-accuracy']
    n_rand_epochs = dataframe['rand-epoch']
    train_loss = dataframe['avg-train-loss']
    j = 0
    styles = [(0, ()), (0, (1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)),
             (0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1)), (5, (10, 3))]
    for i in range(len(n_epochs)):
        style = styles[i]
        if metric == 'acc':
            if i == 0:
                j = i
            else:
                j = k
            k = (i + 1) * len(set(n_het_level))
            ax.plot(n_het_level[j:k], test_acc[j:k], marker=marker, label=label+ ' E= '+ str(n_epochs[i]), linestyle=style)
            plt.xlabel('Level of System Heterogeneity')
            plt.ylabel('Average Test Accuracy')
        elif metric == 'loss':
            if i == 0:
                j = i
            else:
                j = k
            k = (i + 1) * len(set(n_het_level))
            ax.plot(n_het_level[j:k], train_loss[j:k], marker=marker, label=label + ' E= '+ str(n_epochs[i]), linestyle=style)
            plt.xlabel('Level of System Heterogeneity')
            plt.ylabel('Loss')
        if metric == 'acc':
            plt.legend()
            fig.savefig(file_name + f'/sys_hetero_origin_E_{n_epochs[i]}_rand_E_{n_rand_epochs[j]}_avg_accu.png',
                        bbox_inches='tight')
        elif metric == 'loss':
            plt.legend()
            fig.savefig(file_name+ f'/sys_hetero_origin_E_{n_epochs[i]}_rand_E_{n_rand_epochs[j]}_train_loss.png', bbox_inches='tight')


def plotResults(files, algos, num_algs):
    #nb_clients = [3, 5, 7, 10]
    marker = ["D", "o", "v", "p", "*", "d", ',', 'P']
    labels = algos
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    #for each algo(one line) in the results-file-name plot the acc results in one figure
    for i in range(num_algs):
        metric = 'acc'
        plotOneAlg(files[i],metric,marker[i],labels[i],ax, fig)
    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    for i in range(num_algs):
        metric = 'loss'
        plotOneAlg(files[i],metric,marker[i],labels[i],ax2, fig)


if __name__ == '__main__':
    lines = 0

    df = pd.read_csv("results/sys_heterog/name_file_res_algos.csv")
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
