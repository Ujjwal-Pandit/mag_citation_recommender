#!/bin/bash
# Auto-detection installation script for GPU cluster
# This script tries different CUDA versions and detects which one works

set -e

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
echo "Step 2: Detecting installed CUDA version..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"✓ CUDA detected: {cuda_version}")
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    
    # Determine CUDA major.minor version
    if cuda_version:
        cuda_major_minor = '.'.join(cuda_version.split('.')[:2])
        print(f"✓ CUDA version for PyG: {cuda_major_minor}")
        
        # Write to file for next steps
        with open('.cuda_version', 'w') as f:
            f.write(cuda_major_minor)
    else:
        print("⚠ CUDA version string not available")
        with open('.cuda_version', 'w') as f:
            f.write('unknown')
else:
    print("⚠ CUDA not available - using CPU version")
    with open('.cuda_version', 'w') as f:
        f.write('cpu')
EOF

# Step 3: Install PyTorch Geometric dependencies based on detected CUDA
echo ""
echo "Step 3: Installing PyTorch Geometric dependencies..."

CUDA_VER=$(cat .cuda_version 2>/dev/null || echo "cu118")

if [ "$CUDA_VER" = "cpu" ]; then
    echo "Installing CPU versions..."
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html || echo "Warning: torch-scatter installation failed"
    pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html || echo "Warning: torch-sparse installation failed"
    pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html || echo "Warning: torch-cluster installation failed"
elif [ "$CUDA_VER" = "11.8" ] || [ "$CUDA_VER" = "unknown" ]; then
    echo "Installing for CUDA 11.8..."
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html || echo "Warning: torch-scatter installation failed"
    pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html || echo "Warning: torch-sparse installation failed"
    pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html || echo "Warning: torch-cluster installation failed"
elif [ "$CUDA_VER" = "12.1" ] || [ "$CUDA_VER" = "12.2" ] || [ "$CUDA_VER" = "12.3" ] || [ "$CUDA_VER" = "12.4" ]; then
    echo "Installing for CUDA 12.1+..."
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html || echo "Warning: torch-scatter installation failed"
    pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html || echo "Warning: torch-sparse installation failed"
    pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html || echo "Warning: torch-cluster installation failed"
else
    echo "Trying CUDA 11.8 as default..."
    pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html || echo "Warning: torch-scatter installation failed"
    pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html || echo "Warning: torch-sparse installation failed"
    pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html || echo "Warning: torch-cluster installation failed"
fi

# Step 4: Install remaining packages
echo ""
echo "Step 4: Installing remaining packages..."
pip install torch-geometric==2.4.0
pip install sentence-transformers==2.2.2
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install tqdm==4.66.1

# Step 5: Verify installation
echo ""
echo "Step 5: Verifying installation..."
python3 << 'EOF'
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    import torch_geometric
    print(f"✓ torch-geometric: {torch_geometric.__version__}")
    
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers: OK")
    
    import numpy as np
    print(f"✓ numpy: {np.__version__}")
    
    import pandas as pd
    print(f"✓ pandas: {pd.__version__}")
    
    print("\n✅ All packages installed successfully!")
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    exit(1)
EOF

# Cleanup
rm -f .cuda_version

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="

