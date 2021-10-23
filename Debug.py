import torch
import MLlib
import numpy as np
import torch.nn as nn
from MLlib.Trainer import Trainer
import matplotlib.pyplot as plt



X = MLlib.DSP.get_data()


net = MLlib.Models.AutoEncoder(2)

trainer = Trainer(net)
trainer.train(sequence, 1000)

preds = torch.Tensor(sequence)
seq2 = torch.Tensor([np.sin(np.linspace(0,2*np.pi, 100)),
                np.cos(np.linspace(0,2*np.pi, 100))]).T

seq2_rec = net(seq2)
seq2_rec = seq2_rec.detach().numpy()
plt.plot(seq2[:, 0], 'r')
plt.plot(seq2_rec[:, 0], 'b')
plt.show()
