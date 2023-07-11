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
# the assigned values to the param : [100, 200, 300, 400, 500, 600,800, 1000]


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



###############################################################################################################
###############################################################################################################
################################ Run the main file for test server generalisation (w.r.t number of clients)
# the assigned values to the param :  [2, 4, 6, 7, 8, 9, 10]


####### FedAvg
python main_server_general.py --device 'cpu' --alg 'fedavg'  --iters 300 --epochs 5 --batch 32 --non_iid_alpha 0.4

####### FedProx
python main_server_general.py --device 'cpu' --alg 'fedprox'  --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.4 --mu 0.1

####### FedAP
python main_server_general.py --device 'cpu' --alg 'fedap'  --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.4 --model_momentum 0.3

####### FedBN
python main_server_general.py --device 'cpu' --alg 'fedbn'  --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.4

####### Metafed
python main_server_general.py --device 'cpu' --alg 'metafed'  --iters 300  --epochs 5 --batch 32 --non_iid_alpha 0.4 --threshold 1.1 --nosharebn

#***************************** Plot the results (accuracy and loss) of all the algo in one figure
python server_general_plot.py




###############################################################################################################
###############################################################################################################
################################ Run the main file for syste, heterogeneity (w.r.t number of epochs)
# the assigned values to the param :  [0.0, 0.1, 0.3,0.5, 0.9] ######################### n_epochs : 1,3,5


####### FedAvg
python main_sys_heterog.py --device 'cpu' --alg 'fedavg'  --iters 300 --n_clients 10 --batch 32 --non_iid_alpha 0.1

####### FedProx
python main_sys_heterog.py --device 'cpu' --alg 'fedprox'  --iters 300 --n_clients 10  --batch 32 --non_iid_alpha 0.1 --mu 0.1

####### FedAP
python main_sys_heterog.py --device 'cpu' --alg 'fedap'  --iters 300 --n_clients 10  --batch 32 --non_iid_alpha 0.1 --model_momentum 0.3

####### FedBN
python main_sys_heterog.py --device 'cpu' --alg 'fedbn'  --iters 300  --n_clients 10  --batch 32 --non_iid_alpha 0.1

####### Metafed
python main_sys_heterog.py --device 'cpu' --alg 'metafed'  --iters 300  --n_clients 10  --batch 32 --non_iid_alpha 0.1 --threshold 1.1 --nosharebn

#***************************** Plot the results (accuracy and loss) of all the algo in one figure
python sys_heterog_plot.py




###############################################################################################################
###############################################################################################################
################################ Run the main file for convergence rate test (w.r.t number of rounds)
# the assigned values to the param :  [100, 200, 300, 400,500,600] ################## n_epochs : 1,3,5; batch 16,32,64


####### FedAvg
python main_conv_rate.py --device 'cpu' --alg 'fedavg'  --n_clients 10 --non_iid_alpha 0.1

####### FedProx
python main_conv_rate.py --device 'cpu' --alg 'fedprox' --n_clients 10 --non_iid_alpha 0.1 --mu 0.1

####### FedAP
python main_conv_rate.py --device 'cpu' --alg 'fedap' --n_clients 10  --non_iid_alpha 0.1 --model_momentum 0.3

####### FedBN
python main_conv_rate.py --device 'cpu' --alg 'fedbn' --n_clients 10 --non_iid_alpha 0.1

####### Metafed
python main_conv_rate.py --device 'cpu' --alg 'metafed' --n_clients 10 --non_iid_alpha 0.1 --threshold 1.1 --nosharebn

#***************************** Plot the results (accuracy and loss) of all the algo in one figure
python conv_rate_plot.py





