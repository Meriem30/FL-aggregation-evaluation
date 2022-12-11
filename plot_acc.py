# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 08:11:16 2022

@author: TOSHIBA-Portégé C30
"""
from matplotlib import pyplot as plt

import pandas
df = pandas.read_csv('acuuracy_results.csv')
print(df)

#nbClients = df['num_clients']
nbClients = [3,5,10,20]
test_acc_fedavg_nbClient = df['avg-test-accuracy']
test_acc_fedprox_nbClient = df['avg-test-accuracy']*2
test_acc_fedben_nbClient = df['avg-test-accuracy']*3


fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
#ax2 = fig.add_subplot(2, 2, 2)
#ax3 = fig.add_subplot(2, 2, 3)


ax1.plot(nbClients, test_acc_fedavg_nbClient, label='FedAvg')
ax1.plot(nbClients, test_acc_fedprox_nbClient, label='FedProx')
ax1.plot(nbClients, test_acc_fedben_nbClient, label='FedBen')
plt.legend()
#ax2.plot(nbClients, test_acc_fedprox_nbClient, label='FedProx')
#ax2.plot(nbClients, test_acc_fedben_nbClient, label='FedBen')


ax1.set_xlabel('Number Of Clients')
ax1.set_ylabel('Test Accuracy')
ax1.set_title('Performance test based on Number of Clients')

plt.show()

fig.savefig('Num of Clients.png', bbox_inches='tight')

plt.close(fig)