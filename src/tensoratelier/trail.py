import torch
import torch.functional as F
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv

import tensoratelier as ta


class LitAutoEncoder(ta.AtelierModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = tv.datasets.MNIST(
    ".", download=True, transform=tv.transforms.ToTensor())

train, val = data.random_split(dataset, [55000, 5000])

autoencoder = LitAutoEncoder()
trainer = ta.AtelierTrainer()
trainer.fit(autoencoder, data.DataLoader(train), data.DataLoader(val))
