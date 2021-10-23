from numpy.core.fromnumeric import size
import torch
from torch._C import INSERT_FOLD_PREPACK_OPS
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc3 = nn.Conv1d(88, 40, 1000, stride=2)
        self.pool3 = nn.MaxPool1d(2)
        self.fc4 = nn.Conv1d(40, 10, 1000, stride=2)
        self.pool4 = nn.MaxPool1d(2)
        self.fc5 = nn.Conv1d(10, 4, 50)
        self.fc6 = nn.Linear(1056, 50)
            
    def forward(self, x):
        out = self.fc3(x)
        out = self.pool3(out)
        out = self.fc4(out)
        out = self.pool4(out)
        out = self.fc5(out)
        out = torch.flatten(out, start_dim=1)
        z = self.fc6(out)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(50, 1056)
        self.upsample1 = nn.Upsample(313)
        self.fc2 = nn.ConvTranspose1d(4, 10, 50)
        self.upsample2 = nn.Upsample(626)
        self.fc3 = nn.ConvTranspose1d(10, 40, 1000, stride=2)
        self.upsample3 = nn.Upsample(4501)
        self.fc4 = nn.ConvTranspose1d(40, 88, 1000, stride=2)

    def forward(self, z):
        out = self.fc1(z)
        out = torch.reshape(out,(z.shape[0], 4, 264))
        out = self.fc2(out)
        out = self.upsample2(out)
        out = self.fc3(out)
        out = self.upsample3(out)
        out = self.fc4(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self,):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
