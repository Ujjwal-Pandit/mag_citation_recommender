# How to Check CUDA Version on GPU Server

## Method 1: Using PyTorch (Recommended - No Admin Required)

After installing PyTorch, check the CUDA version:

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

Or more detailed:
```bash
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA not available (may be on login node - request GPU node)")
EOF
```

## Method 2: Using nvidia-smi (If Allowed)

```bash
nvidia-smi
```

This shows:
- CUDA Version (at the top right)
- GPU information
- Driver version

## Method 3: Using nvcc (If CUDA Toolkit Installed)

```bash
nvcc --version
```

Or:
```bash
nvcc -V
```

## Method 4: Check CUDA Library Files

```bash
# Check CUDA library version
cat /usr/local/cuda/version.txt 2>/dev/null || echo "File not found"

# Or check the library directly
ls -la /usr/local/cuda*/version.txt 2>/dev/null
```

## Method 5: Using Python to Check System CUDA

```bash
python3 << 'EOF'
import subprocess
import os

# Try to find CUDA version from common locations
cuda_paths = [
    '/usr/local/cuda/version.txt',
    '/usr/local/cuda/version',
    '/opt/cuda/version.txt'
]

for path in cuda_paths:
    if os.path.exists(path):
        with open(path, 'r') as f:
            print(f"CUDA version from {path}:")
            print(f.read())
            break
else:
    print("CUDA version file not found in common locations")
    print("Try installing PyTorch and checking with: python3 -c \"import torch; print(torch.version.cuda)\"")
EOF
```

## Method 6: Check CUDA in Conda Environment

If using conda with CUDA packages:

```bash
conda list | grep cuda
```

## Quick Check Script

Save this as `check_cuda.sh`:

```bash
#!/bin/bash
echo "=== CUDA Version Check ==="
echo ""

echo "1. PyTorch CUDA (if installed):"
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "  PyTorch not installed"

echo ""
echo "2. nvidia-smi:"
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available or not allowed"

echo ""
echo "3. nvcc:"
nvcc --version 2>/dev/null | grep "release" || echo "  nvcc not found"

echo ""
echo "4. CUDA version file:"
cat /usr/local/cuda/version.txt 2>/dev/null || echo "  CUDA version file not found"
```

Make it executable and run:
```bash
chmod +x check_cuda.sh
./check_cuda.sh
```

## Important Notes

1. **Login Node vs Compute Node**: 
   - Login nodes often don't have GPU access
   - CUDA may show as "not available" on login nodes
   - Request GPU node: `srun --gres=gpu:1 --pty bash`

2. **PyTorch CUDA Build vs System CUDA**:
   - PyTorch is built with a specific CUDA version
   - Check with: `python3 -c "import torch; print(torch.__version__)"`
   - Look for `+cu118` or `+cu121` in the version string

3. **Runtime vs Driver**:
   - `nvidia-smi` shows driver version (what the system supports)
   - `nvcc --version` shows toolkit version (what's installed)
   - PyTorch shows runtime version (what PyTorch was compiled with)

## Recommended Approach

Since you can't use `nvidia-smi`, the best approach is:

1. Install PyTorch with CUDA support
2. Check the CUDA version programmatically:
   ```bash
   python3 -c "import torch; print(torch.version.cuda)"
   ```
3. This tells you what CUDA version PyTorch was built for
4. Install PyTorch Geometric dependencies matching that version

