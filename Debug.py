import torch
import MLlib
import numpy as np



sequence = np.arange(100)
#sequence[40:45] = 0
#sequence[95:100] = 0

net = MLlib.Models.AutoEncoder(1,2)

