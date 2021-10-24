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


def parse_synthesized(synth):
    x = np.zeros_like(synth, dtype=int)
    mask = synth < np.random.uniform(size=x.shape)
    x[mask] = 100
    return x

def merge_midi_matrix(part1, part2):
    net = MLlib.Models.Net()
    net.load_state_dict(torch.load("trained_models_with_logits/sample130.pt", map_location='cpu'))
    net.eval()

    priming_matrix = torch.Tensor(part1)
    priming_matrix = priming_matrix.reshape((priming_matrix.shape[0], 1, priming_matrix.shape[1]))
    print(priming_matrix.shape)
    synthesized = net.synthesise(priming_matrix, 2000)
    synthesized = synthesized[:, 0, :]
    print(synthesized)
    out = np.vstack((part1, synthesized))
    out = np.vstack((out, part2))

    return out.astype('int')
