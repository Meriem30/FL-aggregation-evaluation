#!/bin/sh

###############################################################################################################
###############################################################################################################
###############################run the main file for different number of clients
# the assigned values of the param: [2,5,8,10]


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


###############################################################################################################
###############################################################################################################
###############################run the main file for different number of iterations (rounds)
# the assigned values to the param : [100, 200, 300, 500, 800,1000]


####### FedAvg
python main_n_rounds.py --device 'cpu' --alg 'fedavg' --n_clients 10 --epochs 5 --batch 32 --non_iid_alpha 0.1

####### FedProx
python main_n_rounds.py --device 'cpu' --alg 'fedprox' --n_clients 10 --epochs 5 --batch 32 --non_iid_alpha 0.1 --mu 0.1

####### FedAP
python main_n_rounds.py --device 'cpu' --alg 'fedap' --n_clients 10 --epochs 5 --batch 32 --non_iid_alpha 0.1 --model_momentum 0.3

####### FedBN
python main_n_rounds.py --device 'cpu' --alg 'fedbn' --n_clients 10 --epochs 5 --batch 32 --non_iid_alpha 0.1

####### Metafed
python main_n_rounds.py --device 'cpu' --alg 'metafed' --n_clients 10 --epochs 5 --batch 32 --non_iid_alpha 0.1 --threshold 1.1 --nosharebn

#***************************** Plot the results (accuracy and loss) of all the algo in one figure
python n_rounds_plot_acc.py





###############################################################################################################
###############################################################################################################
################################ Run the main file for different percentage of clients drop-out (unavailability)
# assign values to the parameter : [0.0, 0.1,0.2,0.5,0.9]

####### FedAvg
python main_n_drop.py --device 'cpu' --alg 'fedavg' --n_clients 10 --iters 300 --epochs 5 --batch 32 --non_iid_alpha 0.1

####### FedProx
python main_n_drop.py --device 'cpu' --alg 'fedprox' --n_clients 10 --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.1 --mu 0.1

####### FedAP
python main_n_drop.py --device 'cpu' --alg 'fedap' --n_clients 10 --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.1 --model_momentum 0.3

####### FedBN
python main_n_drop.py --device 'cpu' --alg 'fedbn' --n_clients 10 --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.1

####### Metafed
python main_n_drop.py --device 'cpu' --alg 'metafed' --n_clients 10 --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.1 --threshold 1.1 --nosharebn

#***************************** Plot the results (accuracy and loss) of all the algo in one figure
python n_drop_plot_acc.py




###############################################################################################################
###############################################################################################################
################################ Run the main file for different data distribution (non-iid)
# the assigned values to the param :  [0.0, 0.1, 0.2, 0.5, 0.7, 0.9]


####### FedAvg
python main_n_data_dist.py --device 'cpu' --alg 'fedavg' --n_clients 10 --iters 300 --epochs 5 --batch 32 --non_iid_alpha 0.1

####### FedProx
python main_n_data_dist.py --device 'cpu' --alg 'fedprox' --n_clients 10 --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.1 --mu 0.1

####### FedAP
python main_n_data_dist.py --device 'cpu' --alg 'fedap' --n_clients 10 --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.1 --model_momentum 0.3

####### FedBN
python main_n_data_dist.py --device 'cpu' --alg 'fedbn' --n_clients 10 --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.1

####### Metafed
python main_n_data_dist.py --device 'cpu' --alg 'metafed' --n_clients 10 --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.1 --threshold 1.1 --nosharebn

#***************************** Plot the results (accuracy and loss) of all the algo in one figure
python n_data_dist_plot.py


