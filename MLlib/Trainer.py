import torch
import torch.nn as nn
import torch.optim as optim

class Trainer():
    def __init__(self, model:nn.Module) -> None:
        self.optim = optim.SGD(model.parameters(), 1e-1)
        self.model:nn.Module = model
        self._loss_func = nn.MSELoss()

        

    def train(self, data, epochs, data_distorted, data_chunk_removed):
        reconstruction_error = []
        denoising_error = []
        for ep in range(epochs):
            preds = self.model.forward(data)
            loss = self._loss_func(preds, data)/(data.shape[0]*data.shape[1])
            loss.backward()
            print(loss)

            self.optim.step()
            self.optim.zero_grad()

            # print the reconstruction error
            data_denoised = self.model.forward(data_distorted)
            data_reconstructed = self.model.forward(data_chunk_removed)
            reconstruction_loss = float(torch.norm(data-data_reconstructed)**2/(data.shape[0]*data.shape[1]))
            noise_loss = float(torch.norm(data - data_denoised)**2/(data.shape[0]*data.shape[1]))
            
            reconstruction_error.append(reconstruction_loss)
            denoising_error.append(noise_loss)
        return reconstruction_error, denoising_error