from os import remove
import torch
import MLlib
import numpy as np
import torch.nn as nn
from MLlib.Trainer import Trainer
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.nn as nn
import MLlib.DSP as gd 
import sys


net = MLlib.Models.AutoEncoder()
#net.state_dict(torch.load("AEmodel.pt"))

for param in net.parameters():
    print(param.shape)
trainer = Trainer(net)

def noise(x, p=0.01):
    mask = torch.rand(x.shape) < p
    out = deepcopy(x)
    out[mask] = 1 - x[mask]
    return out

def remove_chunk(x, start, T, end=10000):
    out = deepcopy(x)
    end = min(end, start + T)
    slice1 = torch.Tensor([out[0,:, :start]])
    slice2 = torch.rand((1, 88, end-start))
    slice3 = torch.Tensor([out[0, :, end:]])
    out = torch.cat((slice1, slice2, slice3), dim=2)
    return out

def parse_synthesized(synth):
    x = np.zeros_like(synth, dtype=int)
    mask = synth < np.random.uniform(size=x.shape)
    x[mask] = 100
    return x

reconstruction_error = []
denoising_error = []
iter = 0

for x in gd.get_data('.'):
    N = x.shape[0]-1
    x_inputs = x[:N]
    y_inputs = x[1:]
    x_inputs = torch.Tensor(x_inputs).reshape((N, 1, 88)).to("cuda")
    y_inputs = torch.Tensor(y_inputs).reshape((N, 1, 88)).to("cuda")
    #y, (h1, c1), (h2, c2) = net.forward(x)
    x_disturbed = noise(x)
    #x_with_hole = remove_chunk(x, 5000, 1000)


    rec_error, denoise_error = trainer.train_AE(x_inputs, 5, x_disturbed)
    #reconstruction_error.extend(rec_error)
    denoising_error.extend(denoise_error)
    # print(gd.arry2mid(x))  
    # dada1 = x
    # data2 = gd.arry2mid(data1)
    # print(data1)
    # print(data2)
    iter += 1
    if iter >= 50:
        break
    

torch.save(net.state_dict(), "AEmodel.pt")
