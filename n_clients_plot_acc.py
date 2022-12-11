# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 08:11:16 2022

@author: TOSHIBA-Portégé C30
"""
from matplotlib import pyplot as plt

import pandas

def plotResults(file_name_fedavg, file_name_fedprox, file_name_fedbn, file_name_fedap, file_name_metafed):
    df_fedavg = pandas.read_csv(file_name_fedavg)
    df_fedprox = pandas.read_csv(file_name_fedprox)
    df_fedbn = pandas.read_csv(file_name_fedbn)
    df_fedap = pandas.read_csv(file_name_fedap)
    df_metafed = pandas.read_csv(file_name_metafed)
    #print(df)
    
    nbClients = [5,10,15,20]
    
    test_acc_fedavg_nbClient = df_fedavg['avg-test-accuracy']
    test_acc_fedprox_nbClient = df_fedprox['avg-test-accuracy']
    test_acc_fedben_nbClient = df_fedbn['avg-test-accuracy']
    test_acc_fedap_nbClient = df_fedap['avg-test-accuracy']
    test_acc_metafed_nbClient = df_metafed['avg-test-accuracy']
    
    
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1, 1, 1)
    #ax2 = fig.add_subplot(2, 2, 2)
    #ax3 = fig.add_subplot(2, 2, 3)
    
    
    ax1.plot(nbClients, test_acc_fedavg_nbClient, label='FedAvg')
    ax1.plot(nbClients, test_acc_fedprox_nbClient, label='FedProx')
    ax1.plot(nbClients, test_acc_fedben_nbClient, label='FedBen')
    ax1.plot(nbClients, test_acc_fedap_nbClient, label='Fedap')
    ax1.plot(nbClients, test_acc_metafed_nbClient, label='Metafed')
    plt.legend()
    #ax2.plot(nbClients, test_acc_fedprox_nbClient, label='FedProx')
    #ax2.plot(nbClients, test_acc_fedben_nbClient, label='FedBen')
    
    
    ax1.set_xlabel('Number Of Clients')
    ax1.set_ylabel('Test Accuracy')
    #ax1.set_title('Performance test based on Number of Clients')
    
    plt.show()
    
    fig.savefig('N_Clients_Accu.png', bbox_inches='tight')
    
    plt.close(fig)