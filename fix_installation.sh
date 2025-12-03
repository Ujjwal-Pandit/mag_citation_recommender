#!/bin/bash
# Fix script for existing installation issues
# Run this if you encountered errors during initial installation

echo "=========================================="
echo "Fixing Installation Issues"
echo "=========================================="
echo ""

# Step 1: Fix NumPy version (compatible with PyTorch 2.1.0 and mkl)
echo "Step 1: Fixing NumPy version..."
pip install "numpy>=1.24.3,<2.0" --force-reinstall

# Step 2: Fix transformers version (compatible with PyTorch 2.1.0)
echo ""
echo "Step 2: Fixing transformers version..."
pip install "transformers>=4.21.0,<4.36.0" --force-reinstall

# Step 3: Reinstall torch-sparse and torch-cluster with correct CUDA version
echo ""
echo "Step 3: Reinstalling PyTorch Geometric dependencies..."

# Detect PyTorch CUDA build
python3 << 'EOF'
import torch
torch_version = torch.__version__
if '+cu118' in torch_version:
    with open('.cuda_build', 'w') as f:
        f.write('cu118')
elif '+cu121' in torch_version:
    with open('.cuda_build', 'w') as f:
        f.write('cu121')
else:
    with open('.cuda_build', 'w') as f:
        f.write('cu118')  # default
EOF

CUDA_BUILD=$(cat .cuda_build 2>/dev/null || echo "cu118")
echo "Detected PyTorch CUDA build: $CUDA_BUILD"

echo "Installing torch-sparse and torch-cluster..."
pip uninstall -y torch-sparse torch-cluster 2>/dev/null || true
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+${CUDA_BUILD}.html
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+${CUDA_BUILD}.html

# Step 4: Verify
echo ""
echo "Step 4: Verifying fixes..."
python3 << 'EOF'
import sys
errors = []

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except Exception as e:
    errors.append(f"PyTorch: {e}")

try:
    import numpy as np
    print(f"✓ numpy: {np.__version__}")
    if np.__version__.startswith('2.'):
        print("  ⚠ Warning: NumPy 2.x detected - may cause issues")
except Exception as e:
    errors.append(f"numpy: {e}")

try:
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
except Exception as e:
    errors.append(f"transformers: {e}")

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

if errors:
    print(f"\n⚠ Some issues remain:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("\n✅ Installation fixed!")
EOF

rm -f .cuda_build

echo ""
echo "=========================================="
echo "Fix complete!"
echo "=========================================="

