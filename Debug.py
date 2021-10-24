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
import mido

net = MLlib.Models.Net()
net.load_state_dict(torch.load("trained_models_with_logits/sample130.pt", map_location='cpu'))
net.eval()


for param in net.parameters():
    print(param.shape)
# trainer = Trainer(net)

def noise(x, p=0.01):
    mask = torch.rand(x.shape) < p
    out = deepcopy(x)
    out[mask] = 1 - x[mask]
    return out

def remove_chunk(x, start, T, end=10000):
    if type(x) == torch.Tensor:
        out = deepcopy(x)
        end = min(end, start + T)
        slice1 = torch.Tensor([out[0,:, :start]])
        slice2 = torch.rand((1, 88, end-start))
        slice3 = torch.Tensor([out[0, :, end:]])
        out = torch.cat((slice1, slice2, slice3), dim=2)
        return out
    elif type(x) == np.ndarray:
        out = deepcopy(x)
        end = min(end, start + T)
        print(start, end)
        x[start:end] = np.zeros((end-start, 88))
        return x

def parse_synthesized(synth):
    x = np.zeros_like(synth, dtype=int)
    mask = synth < np.random.uniform(size=x.shape)
    x[mask] = 100
    return x

reconstruction_error = []
denoising_error = []
iter = 0

mid = mido.MidiFile("faded.mid")
print(mid)
faded_array = MLlib.DSP.mid2arry(mid)
"""plt.plot(range(faded_array.shape[0]), np.multiply(np.where(faded_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
plt.title("Faded, by Alan Walker")
plt.show()
"""

faded_chunked = remove_chunk(faded_array, 4000, 2000)
"""plt.plot(range(faded_chunked.shape[0]), np.multiply(np.where(faded_chunked>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
plt.title("Faded, by Alan Walker")
plt.show()"""

faded_priming = torch.Tensor(faded_array[:4000])
faded_priming = faded_priming.reshape((faded_priming.shape[0], 1, faded_priming.shape[1]))
print(faded_priming.shape)
synthesized = net.synthesise(faded_priming, 2000)
synthesized = synthesized[:, 0, :]
print(synthesized)
faded_restored = faded_chunked
faded_restored[4000:6000] = synthesized

restored_midi = MLlib.DSP.arry2mid(faded_restored, tempo=666667)
restored_midi.save('./out.mid')

plt.plot(range(faded_restored.shape[0]), np.multiply(np.where(faded_restored>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
plt.title("Faded, by Alan Walker")
plt.show()

out = MLlib.Merger.merge_midi_matrix(faded_array, faded_array)

restored_midi = MLlib.DSP.arry2mid(out, tempo=666667)
restored_midi.save('./out2.mid')
