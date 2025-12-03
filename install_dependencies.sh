#!/bin/bash
# Auto-detection installation script for GPU cluster
# This script tries different CUDA versions and detects which one works

# Don't exit on error - we want to try alternatives
set +e

echo "=========================================="
echo "MAG Citation Recommender - Dependency Installer"
echo "=========================================="
echo ""

# Step 1: Install PyTorch - try different CUDA versions
echo "Step 1: Installing PyTorch..."
echo "Trying CUDA 11.8 first (most common)..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 || {
    echo "CUDA 11.8 failed, trying CUDA 12.1..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 || {
        echo "CUDA versions failed, installing CPU version..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    }
}

# Step 2: Detect which CUDA version was installed
echo ""
echo "Step 2: Detecting installed PyTorch and CUDA version..."
python3 << 'EOF'
import torch
import sys

# Check PyTorch version string to determine CUDA support
torch_version = torch.__version__
print(f"✓ PyTorch version: {torch_version}")

# Check if PyTorch was built with CUDA (even if not available on this node)
if '+cu' in torch_version:
    # Extract CUDA version from PyTorch build string
    if '+cu118' in torch_version:
        cuda_build = '11.8'
        print(f"✓ PyTorch built with CUDA 11.8 support")
    elif '+cu121' in torch_version:
        cuda_build = '12.1'
        print(f"✓ PyTorch built with CUDA 12.1 support")
    else:
        cuda_build = '11.8'  # default
        print(f"✓ PyTorch built with CUDA support (assuming 11.8)")
else:
    cuda_build = 'cpu'
    print(f"✓ PyTorch built for CPU")

# Check if CUDA is actually available (may not be on login nodes)
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"✓ CUDA runtime available: {cuda_version}")
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CUDA runtime not available (you may be on a login node)")
    print("⚠ GPU will be available when you run on a compute node with GPU access")

# Write CUDA build version for PyG installation
with open('.cuda_version', 'w') as f:
    f.write(cuda_build)
    
# Also write whether CUDA is available
with open('.cuda_available', 'w') as f:
    f.write('yes' if torch.cuda.is_available() else 'no')
EOF

# Step 3: Install PyTorch Geometric dependencies based on PyTorch build
echo ""
echo "Step 3: Installing PyTorch Geometric dependencies..."

CUDA_VER=$(cat .cuda_version 2>/dev/null || echo "11.8")

if [ "$CUDA_VER" = "cpu" ]; then
    echo "Installing CPU versions (pre-built wheels)..."
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
elif [ "$CUDA_VER" = "11.8" ] || [ "$CUDA_VER" = "unknown" ]; then
    echo "Installing for CUDA 11.8 (pre-built wheels)..."
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
elif [ "$CUDA_VER" = "12.1" ] || [ "$CUDA_VER" = "12.2" ] || [ "$CUDA_VER" = "12.3" ] || [ "$CUDA_VER" = "12.4" ]; then
    echo "Installing for CUDA 12.1+ (pre-built wheels)..."
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
else
    echo "Trying CUDA 11.8 as default (pre-built wheels)..."
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
fi

# Step 4: Fix NumPy version first (compatibility with PyTorch 2.1.0 and mkl packages)
echo ""
echo "Step 4: Installing compatible NumPy version..."
pip install "numpy>=1.24.3,<2.0" --force-reinstall

# Step 5: Install remaining packages
echo ""
echo "Step 5: Installing remaining packages..."
pip install torch-geometric==2.4.0

# Install compatible transformers version for PyTorch 2.1.0
echo "Installing compatible transformers version..."
pip install "transformers>=4.21.0,<4.36.0"

# Install sentence-transformers (will use compatible transformers)
pip install sentence-transformers==2.2.2

pip install pandas==2.0.3
pip install tqdm==4.66.1

# Step 6: Verify installation
echo ""
echo "Step 6: Verifying installation..."
python3 << 'EOF'
import sys
errors = []

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available (normal if on login node - will work on GPU compute node)")
except Exception as e:
    errors.append(f"PyTorch: {e}")

try:
    import torch_geometric
    print(f"✓ torch-geometric: {torch_geometric.__version__}")
except Exception as e:
    errors.append(f"torch-geometric: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers: OK")
except Exception as e:
    errors.append(f"sentence-transformers: {e}")

try:
    import numpy as np
    print(f"✓ numpy: {np.__version__}")
except Exception as e:
    errors.append(f"numpy: {e}")

try:
    import pandas as pd
    print(f"✓ pandas: {pd.__version__}")
except Exception as e:
    errors.append(f"pandas: {e}")

if errors:
    print(f"\n⚠ Some packages had issues:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("\n✅ All packages installed successfully!")
    print("\n📝 Note: If CUDA is not available, you're likely on a login node.")
    print("   Request GPU resources with: srun --gres=gpu:1 --pty bash")
EOF

# Cleanup
rm -f .cuda_version .cuda_available

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="

