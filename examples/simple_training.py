"""This example demonstrates how to:
1. Create a simple model inheriting from AtelierModule
2. Set up a trainer with profiling
3. Train the model with automatic optimization"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tensoratelier import AtelierModule, AtelierTrainer


class SimpleLinearModel(AtelierModule):

    def __init__(self, input_size: int = 10, output_size: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def create_dummy_data(num_samples: int = 1000, input_size: int = 10):
    # Generate random input data
    x = torch.randn(num_samples, input_size)

    weights = torch.randn(input_size, 1)
    y = x @ weights + torch.randn(num_samples, 1) * 0.1

    return x, y


def main():
    print("Tensor Atelier - Simple Training Example")
    print("=" * 50)

    print("Creating dummy data...")
    x, y = create_dummy_data(num_samples=1000, input_size=10)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Creating model...")
    model = SimpleLinearModel(input_size=10, output_size=1)

    print("Setting up trainer...")
    trainer = AtelierTrainer(
        max_epochs=5,
        accelerator="cpu",
    )

    print("Starting training...")
    trainer.fit(model, dataloader)

    print("Training completed!")

    print("Testing model...")
    model.eval()
    with torch.no_grad():
        test_x = torch.randn(10, 10)
        test_y = model.linear(test_x)
        print(f"Test output shape: {test_y.shape}")
        print(f"Sample predictions: {test_y[:3].flatten()}")


if __name__ == "__main__":
    main()
