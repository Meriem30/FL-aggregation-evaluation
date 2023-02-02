#!/bin/sh

###############################run the main file for different number of clients
####### FedAvg
python main_n_clients.py --device 'cpu' --alg 'fedavg' --iters 300 --epochs 5 --batch 64 --non_iid_alpha 0.1

####### FedProx
python main_n_clients.py --device 'cpu' --alg 'fedprox' --iters 300 --epochs 5 --batch 64 --non_iid_alpha 0.1 --mu 0.1
--model_momentum 0.3

####### FedAP
python main_n_clients.py --device 'cpu' --alg 'fedap' --iters 300 --epochs 5 --batch 64 --non_iid_alpha 0.1 --model_momentum 0.3

####### FedBN
python main_n_clients.py --device 'cpu' --alg 'fedbn' --iters 300 --epochs 5 --batch 64 --non_iid_alpha 0.1

####### Metafed
python main_n_clients.py --device 'cpu' --alg 'metafed' --iters 300 --epochs 5 --batch 64 --non_iid_alpha 0.1 --threshold 1.1 --nosharebn

#***************************** Plot the results (accuracy and loss) of all the algo in one figure
python n_client_plot_acc.py