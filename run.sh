#!/bin/sh

###############################run the main file for different number of clients
# the assigned values of the param: [2,5,8,10]


####### FedAvg
python main_n_clients.py --device 'cuda' --alg 'fedavg' --iters 250 --epochs 5 --batch 16 --lr 0.001
--dataset 'cifar10' --balance True --partition_data 'iid' --download True --preprocess True --verbose True


####### FedDyn
python main_n_clients.py --device 'cuda' --alg 'feddyn' ---iters 250 --epochs 5 --batch 16 --lr 0.001
--dataset 'cifar10' --balance True --partition_data 'iid' --download True --preprocess True --verbose True


####### Power-of-choice
python main_n_clients.py --device 'cuda' --alg 'powerofchoice' --iters 250 --epochs 5 --batch 16 --lr 0.001 -
-dataset 'cifar10' --balance True --partition_data 'iid' --download True --preprocess True --verbose True

####### FedBN
python main_n_clients.py --device 'cuda' --alg 'fedbn' --iters 250 --epochs 5 --batch 16 --lr 0.001
--dataset 'cifar10' --balance True --partition_data 'iid' --download True --preprocess True --verbose True

####### FedPer
#python main_n_clients.py --device 'cpu' --alg 'fedper' --iters 200 --epochs 3 --batch 32 --lr 0.001
#--dataset 'cifar10' --balance False  --unbalance_sgm 0.3 --partition_data 'dirichlet' --diralpha 0.3

#***************************** Plot the results (accuracy and loss) of all the algo in one figure
python n_client_plot_acc.py


