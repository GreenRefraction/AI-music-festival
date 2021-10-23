import torch
import MLlib
import numpy as np
import torch.nn as nn
from MLlib.Trainer import Trainer
import MLlib.DSP as gd 
import sys


gd.filter_data(sys.argv[1])

# gd.filter_data('./Datasets/0/')

#for x in gd.get_data('./Datasets/0/'): 
    #print(x.shape[0])
    # print(gd.arry2mid(x))  
    # dada1 = x
    # data2 = gd.arry2mid(data1)
    # print(data1)
    # print(data2)


