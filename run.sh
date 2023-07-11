#!/bin/sh

###############################run the main file for different number of clients
# the assigned values of the param: [2,5,8,10]


####### FedAvg
python main_n_clients.py --device 'cpu' --alg 'fedavg' --iters 200 --epochs 3 --batch 32 --lr 0.001
 --dataset 'cifar10' --balance False  --unbalance_sgm 0.3 --partition_data 'dirichlet' --diralpha 0.3 --download True --preprocess True --verbose True


####### FedDyn
python main_n_clients.py --device 'cpu' --alg 'feddyn' --iters 200 --epochs 3 --batch 32 --lr 0.001
 --dataset 'cifar10' --balance False  --unbalance_sgm 0.3 --partition_data 'dirichlet' --diralpha 0.3

####### Power-of-choice
python main_n_clients.py --device 'cpu' --alg 'powerofchoice' --iters 200 --epochs 3 --batch 32 --lr 0.001
 --dataset 'cifar10' --balance False  --unbalance_sgm 0.3 --partition_data 'dirichlet' --diralpha 0.3

####### FedBN
python main_n_clients.py --device 'cpu' --alg 'fedbn' --iters 200 --epochs 3 --batch 32 --lr 0.001
  --dataset 'cifar10' --balance False  --unbalance_sgm 0.3 --partition_data 'dirichlet' --diralpha 0.3
####### FedPer
#python main_n_clients.py --device 'cpu' --alg 'fedper' --iters 200 --epochs 3 --batch 32 --lr 0.001
#--dataset 'cifar10' --balance False  --unbalance_sgm 0.3 --partition_data 'dirichlet' --diralpha 0.3

#***************************** Plot the results (accuracy and loss) of all the algo in one figure
python n_client_plot_acc.py

# python main_n_clients.py --device 'cpu' --alg 'fedavg' --iters 200 --epochs 3 --batch 32 --lr 0.001 --dataset 'cifar10' --balance True --partition_data 'iid' --download True --preprocess True --verbose True



