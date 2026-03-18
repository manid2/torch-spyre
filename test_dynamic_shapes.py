#!/usr/bin/env python3
"""Test script for dynamic shaped tensors on Spyre device."""

import torch

# Define the Spyre device
device = torch.device("spyre")


@torch.compile(dynamic=True)
def f(x):
    return x * x.size()[0]


def test_dynamic_shapes():
    """Test dynamic shapes with different tensor sizes."""
    print("Testing dynamic shaped tensors on Spyre device...")
    
    try:
        # Test with different sizes
        sizes = [10, 20, 30, 40]
        
        for size in sizes:
            print(f"\nTesting with size {size}...")
            x = torch.rand(size, device=device)
            result = f(x)
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {result.shape}")
            print(f"  Success!")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dynamic_shapes()
    exit(0 if success else 1)

# Made with Bob
