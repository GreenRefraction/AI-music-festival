import torch
import torch.nn as nn
import torch.optim as optim

class Trainer():
    def __init__(self, model:nn.Module) -> None:
        self.optim = optim.SGD(model.parameters(), 2e-2)
        self.model:nn.Module = model
        self._loss_func = nn.MSELoss()

        

    def train(self, data, epochs):
        for ep in range(epochs):
            preds = self.model.forward(data)
            loss = self._loss_func(preds, data)
            print(float(loss))
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        