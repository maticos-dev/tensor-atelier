<div align="center">
  <img src="logo.svg" alt="Tensor Atelier Logo" width="120" height="120">
  
  # Tensor Atelier
  
  *A minimalist PyTorch training framework with automatic optimization and profiling*
  
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
</div>

---

## 🎯 What is Tensor Atelier?

Tensor Atelier is a clean, modular PyTorch training framework designed for developers who want powerful ML capabilities without the complexity. Built with automatic optimization, built-in profiling, and a flexible architecture that grows with your needs.

## ✨ Key Features

- **🔄 Automatic Optimization** - Handles gradients, optimization steps, and scheduling automatically
- **📊 Built-in Profiling** - Monitor training performance with custom profiler support  
- **⚡ Multi-Accelerator** - CPU, GPU, and custom accelerator support
- **🧩 Modular Design** - Clean separation of concerns with extensible components
- **📦 Smart DataLoader** - Automatic device placement and train/validation splitting
- **🎨 Type Safe** - Full type hints and mypy support

## 🚀 Quick Start

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tensoratelier import AtelierModule, AtelierTrainer

# Define your model
class MyModel(AtelierModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        return self.loss_fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Create data and trainer
x, y = torch.randn(1000, 10), torch.randn(1000, 1)
dataloader = DataLoader(TensorDataset(x, y), batch_size=32)

trainer = AtelierTrainer(max_epochs=10, accelerator="cpu")
trainer.fit(MyModel(), dataloader)
```

## 🏗️ Architecture

### Core Components

**AtelierModule** - Your model base class
```python
class MyModel(AtelierModule):
    def training_step(self, batch, batch_idx):
        # Define your training logic
        return loss
    
    def configure_optimizers(self):
        # Return your optimizer
        return torch.optim.Adam(self.parameters())
```

**AtelierTrainer** - Training orchestrator
```python
trainer = AtelierTrainer(
    max_epochs=10,
    accelerator="cpu",  # or "cuda", "mps"
    profiler=AtelierBaseProfiler()  # optional
)
```

**AtelierDataLoader** - Enhanced data loading
```python
# Automatic device placement and train/val splitting
dataloader = AtelierDataLoader(
    original_dataloader,
    trainer,
    lengths=[0.8, 0.2],  # 80% train, 20% val
    device="cpu"
)
```

### Profiling System

Create custom profilers to monitor your training:

```python
from tensoratelier.profilers import AtelierBaseProfiler
import time

class TimeProfiler(AtelierBaseProfiler):
    def start(self, desc, **kwargs):
        self.active_profiles[desc] = time.perf_counter()
    
    def stop(self, desc, context, **kwargs):
        elapsed = time.perf_counter() - self.active_profiles[desc]
        print(f"{desc}: {elapsed:.4f}s")

# Use it
trainer = AtelierTrainer(
    max_epochs=10, 
    accelerator="cpu", 
    profiler=TimeProfiler()
)
```

## 📦 Installation

```bash
# From source (recommended for development)
git clone https://github.com/tensor-atelier/tensor-atelier.git
cd tensor-atelier
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## 📚 Examples

- **[Basic Training](examples/simple_training.py)** - Simple linear model training
- **[Custom Profiler](examples/custom_profiler.py)** - Implementing a custom profiler

## 🛠️ Development

```bash
# Run tests
pytest

# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

## 🎨 Design Philosophy

- **Simplicity** - Minimal boilerplate for common tasks
- **Modularity** - Pluggable components for extensibility  
- **Performance** - Efficient training loops with profiling
- **Type Safety** - Full type hints for better development experience

## 🔮 Roadmap

- [ ] Validation loop support
- [ ] Model checkpointing
- [ ] Learning rate scheduling
- [ ] Multi-GPU support
- [ ] Distributed training
- [ ] CLI interface

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) or:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
---

<div align="center">
  <strong>Built with ❤️ for the PyTorch community</strong>
</div>
