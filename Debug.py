import torch
import MLlib
import numpy as np
import torch.nn as nn
from MLlib.Trainer import Trainer
import matplotlib.pyplot as plt
from copy import deepcopy

import MLlib.DSP as gd 


net = MLlib.Models.AutoEncoder()
trainer = Trainer(net)

def noise(x, p=0.01):
    mask = torch.rand(x.shape) < p
    out = deepcopy(x)
    out[mask] = 1 - x[mask]
    return out

def remove_chunk(x, start, T, end=10000):
    out = deepcopy(x)
    end = min(end, start+T)
    out[start:end] = 0
    return out


for x in gd.get_data('./Datasets/0/'): 
    print(x.shape)
    x = torch.Tensor(x)
    x_slice = x[:10000].T

    x_input = x_slice.view((1, 88, 10000))

    x_disturbed = noise(x_input)
    x_with_hole = remove_chunk(x, 5000, 1000)

    y = net(x_input)
    reconstruction_error, denoising_error = trainer.train(x_input, 10, x_disturbed, x_with_hole)
    plt.plot(reconstruction_error, label="reconstruction error")
    plt.plot(denoising_error, label="denoising error")
    plt.legend()
    plt.show()
    #print(gd.arry2mid(x))  
    # dada1 = x
    # data2 = gd.arry2mid(data1)
    # print(data1)
    # print(data2)
    
    break