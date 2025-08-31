#!/usr/bin/env python3
"""
Clean CUDA test without any environment manipulation
"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor creation
    x = torch.ones(10, device='cuda')
    print(f"CUDA tensor created: {x.device}")
else:
    print("CUDA not available - checking why...")
    print(f"CUDA built version: {torch.version.cuda}")