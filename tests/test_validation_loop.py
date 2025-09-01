#!/usr/bin/env python3
"""
Test script for validation loop functionality in Tensor Atelier.
This script tests the validation loop implementation and integration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tensoratelier import AtelierModule, AtelierTrainer, AtelierDataLoader


class ValidationTestModel(AtelierModule):
    """Test model that implements both training and validation steps."""
    
    def __init__(self, input_size=10, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss_fn = nn.MSELoss()
        self.validation_losses = []  # Track validation losses
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = self.loss_fn(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx=0):
        """Custom validation step that tracks losses."""
        x, y = batch
        y_hat = self.linear(x)
        loss = self.loss_fn(y_hat, y)
        self.validation_losses.append(loss.item())
        print(f"Validation batch {batch_idx}: loss = {loss.item():.4f}")
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def create_test_data(num_samples=200, input_size=10):
    """Create test data for training and validation."""
    # Generate random input data
    x = torch.randn(num_samples, input_size)
    
    # Create a simple linear relationship with noise
    weights = torch.randn(input_size, 1)
    y = x @ weights + torch.randn(num_samples, 1) * 0.1
    
    return x, y


def test_validation_loop_basic():
    """Test basic validation loop functionality."""
    print("=" * 60)
    print("Testing Basic Validation Loop")
    print("=" * 60)
    
    # Create test data
    x, y = create_test_data(num_samples=200, input_size=10)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model and trainer
    model = ValidationTestModel(input_size=10, output_size=1)
    trainer = AtelierTrainer(max_epochs=2, accelerator="cpu")
    
    # Train with validation
    print("Starting training with validation...")
    trainer.fit(model, dataloader, train_val_split=[0.8, 0.2])
    
    # Check that validation losses were recorded
    print(f"\nValidation losses recorded: {len(model.validation_losses)}")
    if model.validation_losses:
        print(f"Average validation loss: {sum(model.validation_losses) / len(model.validation_losses):.4f}")
        print(f"First validation loss: {model.validation_losses[0]:.4f}")
        print(f"Last validation loss: {model.validation_losses[-1]:.4f}")
    
    return len(model.validation_losses) > 0


def test_validation_loop_manual():
    """Test validation loop manually without trainer."""
    print("\n" + "=" * 60)
    print("Testing Manual Validation Loop")
    print("=" * 60)
    
    # Create test data
    x, y = create_test_data(num_samples=100, input_size=5)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Create model
    model = ValidationTestModel(input_size=5, output_size=1)
    model.eval()
    
    # Manual validation loop
    print("Running manual validation...")
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            loss = model.validation_step(batch, batch_idx)
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"Manual validation completed:")
        print(f"  Batches processed: {num_batches}")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Total validation losses tracked: {len(model.validation_losses)}")
        return True
    else:
        print("No validation batches processed!")
        return False


def test_validation_without_implementation():
    """Test validation with model that doesn't implement validation_step."""
    print("\n" + "=" * 60)
    print("Testing Validation Without Custom Implementation")
    print("=" * 60)
    
    class SimpleModel(AtelierModule):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 1)
            self.loss_fn = nn.MSELoss()
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.linear(x)
            return self.loss_fn(y_hat, y)
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)
    
    # Create test data
    x, y = create_test_data(num_samples=50, input_size=5)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = SimpleModel()
    model.eval()
    
    # Test default validation_step behavior
    print("Testing default validation_step behavior...")
    batch = next(iter(dataloader))
    result = model.validation_step(batch, 0)
    
    print(f"Default validation_step result: {result}")
    print("✓ Default validation_step returns None (as expected)")
    
    return result is None


def test_validation_loop_integration():
    """Test validation loop integration with AtelierDataLoader."""
    print("\n" + "=" * 60)
    print("Testing Validation Loop Integration")
    print("=" * 60)
    
    # Create test data
    x, y = create_test_data(num_samples=300, input_size=8)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=25, shuffle=True)
    
    # Create model and trainer
    model = ValidationTestModel(input_size=8, output_size=1)
    trainer = AtelierTrainer(max_epochs=1, accelerator="cpu")
    
    # Create AtelierDataLoader with train/val split
    atelier_dataloader = AtelierDataLoader(
        dataloader, 
        trainer, 
        lengths=[0.7, 0.3],  # 70% train, 30% validation
        device="cpu"
    )
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(atelier_dataloader.train_ds)}")
    print(f"Validation dataset size: {len(atelier_dataloader.val_ds)}")
    
    # Train with validation
    print("\nStarting integrated training with validation...")
    trainer.fit(model, atelier_dataloader)
    
    # Check results
    print(f"\nIntegration test results:")
    print(f"  Validation losses recorded: {len(model.validation_losses)}")
    if model.validation_losses:
        print(f"  Average validation loss: {sum(model.validation_losses) / len(model.validation_losses):.4f}")
        return True
    else:
        print("  No validation losses recorded!")
        return False


def main():
    """Run all validation loop tests."""
    print("Tensor Atelier - Validation Loop Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Validation Loop", test_validation_loop_basic),
        ("Manual Validation Loop", test_validation_loop_manual),
        ("Validation Without Implementation", test_validation_without_implementation),
        ("Validation Loop Integration", test_validation_loop_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            result = test_func()
            results.append((test_name, result, None))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ PASSED" if result else ("❌ FAILED" if error is None else "💥 ERROR")
        print(f"{status}: {test_name}")
        if error:
            print(f"    Error: {error}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation loop tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
