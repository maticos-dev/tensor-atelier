#!/usr/bin/env python3
"""
Simple test to debug the dataloader issue.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def test_dataloader():
    print("Testing dataloader...")
    
    # Create simple dataset
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(x, y)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test random_split
    train_length = int(0.8 * len(dataset))
    val_length = len(dataset) - train_length
    lengths = [train_length, val_length]
    
    print(f"Split lengths: {lengths}")
    
    train_dataset, val_dataset = random_split(dataset, lengths)
    
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")
    
    # Test DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print("Testing DataLoader iteration...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}: {batch[0].shape}, {batch[1].shape}")
        if i >= 2:  # Just test first few batches
            break
    
    print("DataLoader test completed successfully!")

if __name__ == "__main__":
    test_dataloader()
