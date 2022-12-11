# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 11:00:42 2022

@author: TOSHIBA-Portégé C30
"""

from alg.fedavg import fedavg

class fedbn(fedavg):
    def __init__(self,args):
        super(fedbn, self).__init__(args)