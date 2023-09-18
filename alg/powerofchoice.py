import random
import numpy as np
from alg.fedavg import fedavg
class powerofchoice(fedavg):

    def __init__(self, args):
        super(powerofchoice, self).__init__(args)

    def sample_condidates(self, args):
        selection = random.sample(range(args.n_clients), args.d)
        selection = sorted(selection)
        return selection

    def sample_clients(self, args, condidates, losses):
        sort = np.array(losses).argsort().tolist()
        #sort.reverse()
        selected_clients = np.array(condidates)[sort][0:args.d]
        return selected_clients.tolist()







