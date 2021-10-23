import torch
from torch._C import INSERT_FOLD_PREPACK_OPS
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Encoder, self).__init__()
        self._in_size = in_size
        self._out_size = out_size

    
    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()
        self._in_size = in_size
        self._out_size = out_size
    
    def forward(self, z):
        pass

class AutoEncoder(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self._in_size = in_size
        self._hidden_size = hidden_size

        self.encoder = Encoder(in_size, hidden_size)
        self.decoder = Decoder(hidden_size, in_size)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out