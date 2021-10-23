import torch
import torch.nn as nn
import torch.optim as optim

class Trainer():
    def __init__(self, model:nn.Module) -> None:
        self.optim = optim.SGD(model.parameters(), 1e-1)
        self.model:nn.Module = model
        self._loss_func = nn.MSELoss()

        

    def train(self, data, epochs, data_distorted):
        reconstruction_error = []
        for ep in range(epochs):
            preds = self.model.forward(data)
            loss = self._loss_func(preds, data)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            # print the reconstruction error
            data_reconstructed = self.model.forward(data_distorted)
            reconstruction_loss = float(torch.norm(data-data_reconstructed))
            reconstruction_error.append(reconstruction_loss)
        return reconstruction_error