#!/usr/bin/env python3
"""
Quick CUDA test script to debug GPU issues
"""

import os
import sys

print("🔍 Testing CUDA setup...")

# Set environment variables like the main app
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"🎮 CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"🔢 CUDA Device Count: {device_count}")
        
        if device_count > 0:
            for i in range(device_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"📱 Device {i}: {gpu_name}")
                    
                    # Test memory
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory / (1024**3)  # GB
                    print(f"💾 Total Memory: {total_memory:.1f} GB")
                    
                    # Test tensor creation
                    test_tensor = torch.ones(1000, 1000).cuda(i)
                    print(f"✅ Successfully created tensor on device {i}")
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"❌ Error with device {i}: {e}")
        else:
            print("❌ No CUDA devices found")
    else:
        print("⚪ Running in CPU-only mode")
        
    print("\n🔍 Environment variables:")
    for key in ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]:
        value = os.environ.get(key, "Not set")
        print(f"  {key}: {value}")
        
    print("\n🔍 System Info:")
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        driver_version = nvml.nvmlSystemGetDriverVersion()
        print(f"🚗 NVIDIA Driver: {driver_version.decode()}")
        
        device_count = nvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle).decode()
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"🎮 GPU {i}: {name}")
            print(f"  💾 Memory: {memory_info.used / (1024**3):.1f}GB / {memory_info.total / (1024**3):.1f}GB")
    except ImportError:
        print("⚠️  nvidia-ml-py not available for detailed GPU info")
    except Exception as e:
        print(f"⚠️  NVIDIA driver info error: {e}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

print("\n✅ CUDA test completed!")