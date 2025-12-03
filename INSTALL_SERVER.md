# Server Installation Guide with Specific Versions

## Step-by-Step Installation

### 1. Clone the repository
```bash
git clone https://github.com/Ujjwal-Pandit/mag_citation_recommender.git
cd mag_citation_recommender
```

### 2. Create Python environment
```bash
# Using conda (recommended)
conda create -n mag_citation python=3.9 -y
conda activate mag_citation

# OR using venv
python3 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with CUDA support

**Option A: Auto-detection script (Recommended - no CUDA version checking needed)**
```bash
# Make script executable and run
chmod +x install_dependencies.sh
./install_dependencies.sh
```

This script will:
- Try installing PyTorch with different CUDA versions automatically
- Detect which CUDA version works after installation
- Install the correct PyTorch Geometric dependencies
- Verify everything is working

**Option B: Manual installation (if you know CUDA version)**

If you cannot check CUDA version with `nvidia-smi`, try these in order:

#### Try CUDA 11.8 first (most common):
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

#### If that fails, try CUDA 12.1:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

#### If both fail, install CPU version:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

**After installing PyTorch, detect the CUDA version programmatically:**
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

### 4. Install PyTorch Geometric dependencies

**If using the auto-detection script (Option A), skip this step - it's handled automatically.**

**If installing manually (Option B), detect CUDA version first:**
```bash
# Detect CUDA version from installed PyTorch
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0]+'.'+torch.version.cuda.split('.')[1] if torch.cuda.is_available() and torch.version.cuda else 'cpu')")
echo "Detected CUDA version: $CUDA_VER"
```

Then install based on detected version:

#### For CUDA 11.8 (or if detection shows 11.x):
```bash
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

#### For CUDA 12.1+ (or if detection shows 12.x):
```bash
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

#### For CPU only:
```bash
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### 5. Install remaining packages
```bash
pip install torch-geometric==2.4.0
pip install sentence-transformers==2.2.2
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install tqdm==4.66.1
```

### 6. Verify installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torch_geometric; print(f'torch-geometric: {torch_geometric.__version__}')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers: OK')"
```

## Alternative: Install from requirements.txt

**Important:** PyTorch must be installed first with the correct CUDA version, then install other packages:

```bash
# Step 1: Install PyTorch (choose based on your CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install PyTorch Geometric dependencies
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Step 3: Install remaining packages from requirements.txt
pip install -r requirements.txt
```

## Complete Package Versions Summary

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.9 | Recommended version |
| torch | 2.1.0 | Install with CUDA support |
| torchvision | 0.16.0 | Install with CUDA support |
| torchaudio | 2.1.0 | Install with CUDA support |
| torch-geometric | 2.4.0 | Graph neural network library |
| torch-scatter | 2.1.2 | Required by torch-geometric |
| torch-sparse | 0.6.17 | Required by torch-geometric |
| torch-cluster | 1.6.1 | Required by torch-geometric |
| sentence-transformers | 2.2.2 | For text embeddings |
| numpy | 1.24.3 | Numerical computing |
| pandas | 2.0.3 | Data manipulation |
| tqdm | 4.66.1 | Progress bars |

## Troubleshooting

### If torch-scatter/sparse/cluster installation fails:
1. Check your PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. Check your CUDA version: `python -c "import torch; print(torch.version.cuda)"`
3. Use the correct wheel URL matching your PyTorch and CUDA versions
4. Visit https://data.pyg.org/whl/ to find the correct wheel files

### If CUDA is not detected:
- Try installing PyTorch with different CUDA versions (11.8, then 12.1)
- After installing PyTorch, check programmatically: `python3 -c "import torch; print(torch.cuda.is_available())"`
- If CUDA is still not available, the cluster may not have GPU access or you may need to request GPU resources via SLURM
- Reinstall PyTorch with the correct CUDA version if needed

## Optional: LLM Integration Packages

If you plan to use the LLM integration section, also install:
```bash
pip install transformers==4.35.2
pip install peft==0.7.1
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.3
pip install huggingface-hub==0.19.4
```

