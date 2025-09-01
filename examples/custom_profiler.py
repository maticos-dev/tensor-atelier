from tensoratelier.profilers import AtelierBaseProfiler
import torch
from tensoratelier import AtelierModule, AtelierTrainer
import torch.nn as nn
import time
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, TensorDataset


class SimpleTimeProfiler(AtelierBaseProfiler):
    """Simple timer-based profiler."""

    def __init__(self, print_results: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.print_results = print_results
        self.timings: Dict[str, list] = {}

    def start(self, desc: str, **kwargs: Any) -> None:
        """Starts timing for the given description."""
        start_time = time.perf_counter()
        if desc not in self.timings:
            self.timings[desc] = []
        self.active_profiles[desc] = start_time

    def stop(self, desc: str, context: Any, **kwargs: Any) -> None:
        """Records elapsed time."""
        if desc in self.active_profiles:
            start_time = self.active_profiles[desc]
            elapsed = time.perf_counter() - start_time
            self.timings[desc].append(elapsed)
            del self.active_profiles[desc]

            if self.print_results:
                print(f"{desc}: {elapsed:.4f}s")

    def get_stats(self) -> Dict[str, Any]:
        """Return timing statistics."""
        stats = {}
        for desc, times in self.timings.items():
            if times:
                stats[desc] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        return stats


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
    x = torch.randn(num_samples, input_size)

    weights = torch.randn(input_size, 1)
    y = x @ weights + torch.randn(num_samples, 1) * 0.1

    return x, y


def main():
    print("Tensor Atelier - Custom Profiler Example")
    print("=" * 50)
    
    profiler = SimpleTimeProfiler(print_results=True)
    
    x, y = create_dummy_data(1000, 10)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    trainer = AtelierTrainer(
        max_epochs=5, 
        accelerator="cpu", 
        profiler=profiler
    )
    
    module = SimpleLinearModel()

    print("Starting training with custom profiler...")
    trainer.fit(module, dataloader)

    # Print final statistics
    print("\nTraining completed!")
    print("Profiling statistics:")
    stats = profiler.get_stats()
    for desc, stat in stats.items():
        print(f"  {desc}:")
        print(f"    Count: {stat['count']}")
        print(f"    Total time: {stat['total_time']:.4f}s")
        print(f"    Average time: {stat['avg_time']:.4f}s")
        print(f"    Min time: {stat['min_time']:.4f}s")
        print(f"    Max time: {stat['max_time']:.4f}s")

    module.eval()
    with torch.no_grad():
        test_x = torch.randn(10, 10)
        test_y = module.linear(test_x)
        print(f"\nTest output shape: {test_y.shape}")
        print(f"Sample predictions: {test_y[:3].flatten()}")


if __name__ == "__main__":
    main()
