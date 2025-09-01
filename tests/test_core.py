import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tensoratelier.core import AtelierModule, AtelierTrainer, AtelierDataLoader, AtelierOptimizer


class SimpleModel(AtelierModule):
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


class TestAtelierModule:
    def test_module_creation(self):
        model = SimpleModel()
        assert isinstance(model, AtelierModule)
        assert isinstance(model, nn.Module)
    
    def test_training_step(self):
        model = SimpleModel()
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        batch = (x, y)
        
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_configure_optimizers(self):
        model = SimpleModel()
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)


class TestAtelierOptimizer:
    def test_optimizer_wrapper(self):
        model = SimpleModel()
        torch_optimizer = torch.optim.Adam(model.parameters())
        atelier_optimizer = AtelierOptimizer(torch_optimizer)
        
        assert atelier_optimizer.step_count == 0
        assert atelier_optimizer.param_groups == torch_optimizer.param_groups


class TestAtelierTrainer:
    def test_trainer_creation(self):
        trainer = AtelierTrainer(
            max_epochs=10,
            accelerator="cpu",
        )
        assert trainer.max_epochs == 10
    
    def test_trainer_validation(self):
        trainer = AtelierTrainer(
            max_epochs=10,
            accelerator="cpu",
        )
        
        model = SimpleModel()
        optimizer = model.configure_optimizers()
        assert trainer.validate_optimizer(optimizer)
        
        atelier_optimizer = AtelierOptimizer(optimizer)
        assert trainer.validate_optimizer(atelier_optimizer)


class TestAtelierDataLoader:
    def test_dataloader_creation(self):
        # Create simple dataset
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=32)
        
        # Create trainer
        trainer = AtelierTrainer(
            max_epochs=10,
            accelerator="cpu",
        )
        
        # Create AtelierDataLoader
        atelier_dataloader = AtelierDataLoader(
            dataloader,
            trainer,
            lengths=[0.8, 0.2],
            device="cpu"
        )
        
        assert atelier_dataloader.trainer == trainer
        assert len(atelier_dataloader.train_ds) == 80  # 80% of 100
        assert len(atelier_dataloader.val_ds) == 20   # 20% of 100
