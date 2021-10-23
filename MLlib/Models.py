import torch
from torch._C import INSERT_FOLD_PREPACK_OPS
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Encoder, self).__init__()
        self._in_size = in_size
        self._out_size = out_size
        self.encoder_hidden_layer = nn.Linear(
            in_features=self._in_size, out_features=100
        )
        self.encoder_output_layer = nn.Linear(
            in_features=100, out_features=200
        )

    
    def forward(self, x):
        activation = self.encoder_hidden_layer(x)
        activation = torch.relu(activation)
        z = self.encoder_output_layer(activation)
        z = torch.relu(z)
        return z


class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()
        self._out_size = out_size
        self.decoder_hidden_layer = nn.Linear(
            in_features=200, out_features=100
        )
        self.decoder_output_layer = nn.Linear(
            in_features=100, out_features=self._out_size
        )

    def forward(self, z):
        activation = self.decoder_hidden_layer(z)
        activation = torch.relu(activation)
        out = self.decoder_output_layer(activation)
        out = torch.relu(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, in_size):
        super(AutoEncoder, self).__init__()
        self._in_size = in_size

        self.encoder = Encoder(in_size, 200)
        self.decoder = Decoder(200, in_size)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
