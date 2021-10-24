import torch
import torch.nn as nn
import torch.optim as optim

def mse_loss(y, y_preds):
    delta = (y- y_preds)
    L = torch.sum(delta*delta)/(delta.shape[0]*delta.shape[1]*delta.shape[2])
    return L

def cross_entropy_loss(y, y_preds):
    L = -torch.sum(y * torch.log(y_preds))/(y.shape[0]*y.shape[1]*y.shape[2])
    return L

class Trainer():
    def __init__(self, model:nn.Module, lr=1e-2) -> None:
        self.optim = optim.SGD(model.parameters(), lr=lr)
        self.model:nn.Module = model
        self._loss_func = cross_entropy_loss
        self._device = "cpu"
        if torch.cuda.is_available():
            self._device = "cuda"
        print(f"device set to {self._device}")

        

    def train(self, data:torch.Tensor, target_data:torch.Tensor, epochs, data_distorted=None, data_chunk_removed=None):
        reconstruction_error = []
        denoising_error = []

        data = data.to(self._device)
        if data_distorted is not None:
            data_distorted = data_distorted.to(self._device)
        if data_chunk_removed is not None:
            data_chunk_removed = data_chunk_removed.to(self._device)
        self.model.train(True)
        self.model.to(self._device)

        for ep in range(epochs):
            preds, (h1, c1) = self.model.forward(data)
            loss = self._loss_func(target_data, preds)
            loss.backward()
            print(float(loss))

            self.optim.step()
            self.optim.zero_grad()

            # print the reconstruction error and denoising error
            if data_distorted is not None:
                data_denoised, _ = self.model.forward(data_distorted)
                noise_loss = float(mse_loss(data, data_denoised))
                denoising_error.append(noise_loss)
            if data_chunk_removed is not None:
                data_reconstructed = self.model.forward(data_chunk_removed)
                reconstruction_loss = float(mse_loss(data, data_reconstructed))
                reconstruction_error.append(reconstruction_loss)
            
        return reconstruction_error, denoising_error

    def train_AE(self, data, epochs, data_distorted=None, data_chunk_removed=None):
        reconstruction_error = []
        denoising_error = []

        data = data.to(self._device)
        if data_distorted is not None:
            data_distorted = data_distorted.to(self._device)
        if data_chunk_removed is not None:
            data_chunk_removed = data_chunk_removed.to(self._device)
        self.model.train(True)
        self.model.to(self._device)

        for ep in range(epochs):
            preds = self.model.forward(data)
            loss = self._loss_func(preds, data)
            loss.backward()
            print(float(loss))

            self.optim.step()
            self.optim.zero_grad()

            # print the reconstruction error and denoising error
            if data_distorted is not None:
                data_denoised = self.model.forward(data_distorted)
                noise_loss = float(mse_loss(data, data_denoised))
                denoising_error.append(noise_loss)
            if data_chunk_removed is not None:
                data_reconstructed = self.model.forward(data_chunk_removed)
                reconstruction_loss = float(mse_loss(data, data_reconstructed))
                reconstruction_error.append(reconstruction_loss)
            
        return reconstruction_error, denoising_error