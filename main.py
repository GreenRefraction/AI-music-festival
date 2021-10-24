<<<<<<< HEAD
=======
from os import remove
>>>>>>> 70490ee7fd6524ca4d50242e8c54110ce5415664
import torch
import MLlib
import numpy as np
import torch.nn as nn
from MLlib.Trainer import Trainer
import matplotlib.pyplot as plt
<<<<<<< HEAD
import MLlib.DSP as gd 


for one_song in gd.get_data('./Datasets/0/'): 
    print(one_song)  
    break


=======
from copy import deepcopy
import torch.nn as nn
import MLlib.DSP as gd 
import sys


net = MLlib.Models.Net()
net.load_state_dict(torch.load("trained_models_with_logits/sample100.pt"))
net.eval()

trainer = Trainer(net, lr=2)

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

for x in gd.get_data('./Datasets/0/'):
    N = x.shape[0]-1
    x_inputs = x[:N]
    y_inputs = x[1:]
    x_inputs = torch.Tensor(x_inputs).reshape((N, 1, 88)).to("cuda")
    y_inputs = torch.Tensor(y_inputs).reshape((N, 1, 88)).to("cuda")
    #y, (h1, c1), (h2, c2) = net.forward(x)
    x_disturbed = noise(x_inputs)
    #x_with_hole = remove_chunk(x, 5000, 1000)


    rec_error, denoise_error = trainer.train(x_inputs, y_inputs, 1, x_disturbed)
    #reconstruction_error.extend(rec_error)
    denoising_error.extend(denoise_error)
    # print(gd.arry2mid(x))  
    # dada1 = x
    # data2 = gd.arry2mid(data1)
    # print(data1)
    # print(data2)
    iter += 1
    if iter % 10 == 0:
        torch.save(net.state_dict(), f"trained_models_with_logits/sample{iter}.pt")
        print(f"trained_models_with_logits/sample{iter}.pt")
        plt.plot(denoising_error)
        plt.savefig("denoising_error.png")
        plt.close()


torch.save(net.state_dict(), "model.pt")
plt.plot(denoising_error)
plt.show()
net.to("cpu")

for x in gd.get_data('./Datasets/0/'):
    synthesized = net.synthesise(torch.Tensor(x).reshape((x.shape[0], 1, 88)), 10000)
    synthesized = synthesized[:, 0, :]
    clean_up_synth = parse_synthesized(synthesized)
    mid = MLlib.DSP.arry2mid(clean_up_synth)
    plt.plot(range(synthesized.shape[0]), np.multiply(np.where(synthesized>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
    plt.title("nocturne_27_2_(c)inoue.mid")
    plt.show()
    mid.save("test.mid")
    break
>>>>>>> 70490ee7fd6524ca4d50242e8c54110ce5415664
