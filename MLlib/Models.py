import torch
import torch.nn as nn
import numpy as np

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.LSTM(88, 200, num_layers=1)
        self.fc2 = nn.LSTM(200, 88, num_layers=1)
        self.fc3 = nn.Sigmoid()

    def forward(self, x:torch.Tensor, h1:torch.Tensor=None, h2=None):
        if h1 is None:
            h10 = torch.zeros(1, x.shape[1], 200).to(x.device)
            c10 = torch.zeros(1, x.shape[1], 200).to(x.device)
        else:
            (h10, c10) = h1
        if h2 is None:
            h20 = torch.zeros(1, x.shape[1], 88).to(x.device)
            c20 = torch.zeros(1, x.shape[1], 88).to(x.device)
        else:
            (h20, c20) = h2
        
        out, (h1n, c1n) = self.fc1(x, (h10, c10))
        out, (h2n, c2n) = self.fc2(out, (h20, c20))
        out = self.fc3(out)
        return out, (h1n, c1n), (h2n, c2n)

    def synthesise(self, x_priming, T):
        out, (h1, c1), (h2, c2) = self.forward(x_priming)
        synthesized = np.zeros((T, 1, 88))
        synthesized[0] = out[-1].detach().numpy()
        for t in range(T-1):
            print(t, end="\r")
            x_in = torch.Tensor(synthesized[t].reshape((1,1,88)))
            out, (h1, c1), (h2, c2) = self.forward(x_in, (h1, c1), (h2,c2))
            p = out[0].detach().numpy()
            sample = np.zeros_like(p)
            mask = p < np.random.uniform(size=p.shape)
            sample[mask] = 1.

        return synthesized
            

    @property
    def InSize(self):
        return self._inSize

    @property
    def Id(self):
        return self._id
        
    @property
    def HiddenSize(self):
        return self._hidden_size