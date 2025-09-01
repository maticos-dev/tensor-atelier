#!/usr/bin/env python3
"""Test script to debug model parameters issue."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tensoratelier import AtelierModule, AtelierTrainer
from tensoratelier.profilers import _FittingProfiler


class SimpleLinearModel(AtelierModule):
    """A simple linear model for demonstration."""
    
    def __init__(self, input_size: int = 10, output_size: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss_fn = nn.MSELoss()
    
    def training_step(self, batch, batch_idx):
        """Define the training step."""
        x, y = batch
        y_hat = self.linear(x)
        loss = self.loss_fn(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=0.001)


def test_model_params():
    print("Testing model parameters...")
    
    # Create model
    model = SimpleLinearModel()
    print(f"Model parameters: {len(list(model.parameters()))}")
    
    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Create trainer
    trainer = AtelierTrainer(max_epochs=1, accelerator="cpu")
    
    # Test training
    trainer.fit(model, dataloader)
    print("Model parameters test completed!")


if __name__ == "__main__":
    test_model_params()
