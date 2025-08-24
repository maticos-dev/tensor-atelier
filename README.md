# Tensor Atelier

A PyTorch training framework with automatic optimization and profiling capabilities.

## Features

- **Automatic Optimization**: Handles gradient computation, optimization steps, and learning rate scheduling automatically
- **Profiling**: Built-in profiling for training and optimization steps
- **Accelerator Support**: Support for different hardware accelerators (CPU, GPU, etc.)
- **Flexible Training Loops**: Customizable training and validation loops
- **DataLoader Integration**: Enhanced DataLoader with automatic device placement and train/validation splitting

## Installation

```bash
pip install tensor-atelier
```

Or install from source:

```bash
git clone https://github.com/tensor-atelier/tensor-atelier.git
cd tensor-atelier
pip install -e .
```

## Quick Start

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tensoratelier import AtelierModule, AtelierTrainer
from tensoratelier.profilers import BaseProfiler

# Define your model
class MyModel(AtelierModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Create data
x = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32)

# Create trainer
trainer = AtelierTrainer(
    max_epochs=10,
    accelerator="cpu",
    profiler=BaseProfiler()
)

# Train
model = MyModel()
trainer.fit(model, dataloader)
```

## Core Components

### AtelierModule

The base class for all models. Inherit from this to create your training models:

```python
class MyModel(AtelierModule):
    def training_step(self, batch, batch_idx):
        # Define your training step
        pass
    
    def configure_optimizers(self):
        # Return your optimizer(s)
        pass
```

### AtelierTrainer

The main training orchestrator:

```python
trainer = AtelierTrainer(
    max_epochs=10,
    accelerator="cpu",  # or "cuda", "mps", etc.
    profiler=BaseProfiler()
)
```

### AtelierDataLoader

Enhanced DataLoader with automatic device placement and train/validation splitting:

```python
dataloader = AtelierDataLoader(
    original_dataloader,
    trainer,
    lengths=[0.8, 0.2],  # 80% train, 20% validation
    device="cpu"
)
```

## Advanced Usage

### Custom Profilers

Create custom profilers by inheriting from `BaseProfiler`:

```python
from tensoratelier.profilers import BaseProfiler

class MyProfiler(BaseProfiler):
    def profile(self, name, *args, **kwargs):
        # Custom profiling logic
        pass
```

### Custom Accelerators

Implement custom accelerators by inheriting from `BaseAccelerator`:

```python
from tensoratelier.accelerators import BaseAccelerator

class MyAccelerator(BaseAccelerator):
    def setup(self, model):
        # Setup logic
        return model
    
    def teardown(self):
        # Cleanup logic
        pass
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/tensor-atelier/tensor-atelier.git
cd tensor-atelier
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by PyTorch Lightning
- Built on top of PyTorch
- Community contributions welcome
