#!/usr/bin/env python3
"""Simple test to debug the dataloader issue."""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def test_dataloader():
    print("Testing dataloader...")
    
    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Test iteration
    for i, (batch_x, batch_y) in enumerate(dataloader):
        print(f"Batch {i}: x shape {batch_x.shape}, y shape {batch_y.shape}")
        if i >= 2:  # Just test first few batches
            break
    
    print("Dataloader test completed!")


if __name__ == "__main__":
    test_dataloader()
