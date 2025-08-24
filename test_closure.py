from tensoratelier.loops.optimization import _AutomaticOptimization
from tensoratelier.core import AtelierTrainer, AtelierModule
from torch.optim import Adam
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys

class MLP(AtelierModule):

	def __init__(self):
		super().__init__()

		self.linear = nn.Linear(10, 1)
		self.loss_fn = nn.MSELoss()

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.linear(x)
		loss = self.loss_fn(y_hat, y)
		return loss

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=0.001)

	@property
	def optimizer(self):
		return self.configure_optimizers()

def create_dummy_data(num_samples: int = 1000, input_size: int = 10):
	    x = torch.randn(num_samples, input_size)
	    weights = torch.randn(input_size, 1)
	    y = x @ weights + torch.randn(num_samples, 1) * 0.1
		
	    return x, y

model = MLP()

trainer = AtelierTrainer(max_epochs=10, accelerator="cpu")

x, y = create_dummy_data(num_samples=1000, input_size=10)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# autoopt = _AutomaticOptimization(trainer, model.optimizer)

trainer.fit(model, dataloader)
