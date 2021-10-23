import torch
import MLlib
import numpy as np
import torch.nn as nn
from MLlib.Trainer import Trainer
import matplotlib.pyplot as plt
from copy import deepcopy

import MLlib.DSP as gd 
import sys


net = MLlib.Models.AutoEncoder()
trainer = Trainer(net)

def noise(x, p=0.01):
    mask = torch.rand(x.shape) < p
    out = deepcopy(x)
    out[mask] = 1 - x[mask]
    return out

def remove_chunk(x, start, T, end=10000):
    out = deepcopy(x).numpy()
    end = min(end, start + T)
    slice1 = torch.Tensor([out[0,:, :start]])
    slice2 = torch.rand((1, 88, end-start))
    slice3 = torch.Tensor([out[0, :, end:]])
    out = torch.cat((slice1, slice2, slice3), dim=2)
    return out


reconstruction_error = []
denoising_error = []
for x in gd.get_training_data('./Datasets/0/', 10000): 
    x = torch.Tensor([x.T])
    print(x.shape)
    x_disturbed = noise(x)
    x_with_hole = remove_chunk(x, 5000, 1000)
    print(x_with_hole.shape)
    rec_error, denoise_error = trainer.train(x, 10, x_disturbed, x_with_hole)
    reconstruction_error.extend(rec_error)
    denoising_error.extend(denoise_error)
    #print(gd.arry2mid(x))  
    # dada1 = x
    # data2 = gd.arry2mid(data1)
    # print(data1)
    # print(data2)

plt.plot(reconstruction_error, label="reconstruction error")
plt.plot(denoising_error, label="denoising error")
plt.legend()
plt.show()