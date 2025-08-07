#!/bin/bash

# GPU Acceleration Installation Script for Olfactory DREAM Model
# This script automatically detects your CUDA version and installs appropriate packages

echo "🚀 Installing GPU acceleration for Olfactory DREAM Model..."
echo "=================================================="

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: nvcc not found. Please install CUDA Toolkit."
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

echo "🔍 Detected CUDA version: $CUDA_VERSION"

# Install appropriate CuPy version
if [ "$CUDA_MAJOR" = "12" ]; then
    echo "📦 Installing CuPy for CUDA 12..."
    pip install cupy-cuda12x>=10.0.0
elif [ "$CUDA_MAJOR" = "11" ]; then
    echo "📦 Installing CuPy for CUDA 11..."
    pip install cupy-cuda11x>=10.0.0
else
    echo "⚠️  Warning: Unsupported CUDA version $CUDA_VERSION"
    echo "Attempting to install CuPy for CUDA 11 (may work)..."
    pip install cupy-cuda11x>=10.0.0
fi

# Install CuML
echo "📦 Installing CuML (RAPIDS)..."
pip install cuml>=22.0.0

# Install other requirements
echo "📦 Installing other requirements..."
pip install -r requirements.txt

# Test installation
echo "🧪 Testing GPU acceleration..."
python3 -c "
try:
    import cupy as cp
    import cuml
    print('✅ CuPy and CuML installed successfully!')
    
    # Test GPU
    device_count = cp.cuda.runtime.getDeviceCount()
    print(f'🎯 Found {device_count} GPU(s)')
    
    if device_count > 0:
        meminfo = cp.cuda.Device().mem_info
        free_mem = meminfo[0] / 1024**3
        total_mem = meminfo[1] / 1024**3
        print(f'📊 GPU Memory: {free_mem:.1f}GB free / {total_mem:.1f}GB total')
        print('🚀 GPU acceleration ready!')
    else:
        print('❌ No GPU devices detected')
        
except ImportError as e:
    print(f'❌ Installation failed: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️  GPU test failed: {e}')
    print('GPU packages installed but GPU may not be accessible')
"

echo "=================================================="
echo "✅ Installation complete!"
echo "Run your model with: python olfaction_model.py --data-dir /path/to/data"
echo "See GPU_SETUP_GUIDE.md for detailed information."