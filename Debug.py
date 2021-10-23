import torch
import MLlib
import numpy as np
import torch.nn as nn
from MLlib.Trainer import Trainer
import matplotlib.pyplot as plt
from copy import deepcopy


X = MLlib.DSP.get_data()
X_distorted = deepcopy(X)
X_distorted[5000:6000] = 0

net = MLlib.Models.AutoEncoder(X.shape[1])

trainer = Trainer(net)
error = trainer.train(X, 100, X_distorted)

plt.plot(error)
plt.show()