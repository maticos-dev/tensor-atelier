from tensoratelier.profilers import BaseProfiler
import torch
from tensoratelier import AtelierModule, AtelierTrainer
import torch.nn as nn
import time
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, TensorDataset


class SimpleTimeProfiler(BaseProfiler):
    """Simple timer-based profiler."""

    def __init__(self, print_results: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.print_results = print_results
        self.timings: Dict[str, list] = {}

    def start(self, desc: str, **kwargs: Any) -> float:
        """Returns start time."""
        start_time = time.perf_counter()
        if desc not in self.timings:
            self.timings[desc] = []
        return start_time

    def stop(self, desc: str, context: float, **kwargs: Any) -> None:
        """Records elapsed time."""
        elapsed = time.perf_counter() - context
        self.timings[desc].append(elapsed)

        if self.print_results:
            print(f"{desc}: {elapsed:.4f}s")


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
    x, y = create_dummy_data(1000, 10)
    trainer = AtelierTrainer(
        max_epochs=5, accelerator="cpu", profiler=SimpleTimeProfiler())
    module = SimpleLinearModel()

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    trainer.fit(module, dataloader)

    module.eval()
    with torch.no_grad():
        test_x = torch.randn(10, 10)
        test_y = module.linear(test_x)
        print(f"Test output shape: {test_y.shape}")
        print(f"Sample predictions: {test_y[:3].flatten()}")


if __name__ == "__main__":
    main()
