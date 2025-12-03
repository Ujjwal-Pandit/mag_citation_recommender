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

**First, check your CUDA version:**
```bash
nvidia-smi
```

**Then install PyTorch based on your CUDA version:**

#### For CUDA 11.8:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1:
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU only (not recommended for training):
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

### 4. Install PyTorch Geometric dependencies
```bash
# Install these in order (they have specific build requirements)
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

**Note:** Replace `cu118` with `cu121` if using CUDA 12.1, or `cpu` for CPU-only builds.

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
- Verify GPU is available: `nvidia-smi`
- Reinstall PyTorch with the correct CUDA version
- Check that CUDA drivers are installed on the system

## Optional: LLM Integration Packages

If you plan to use the LLM integration section, also install:
```bash
pip install transformers==4.35.2
pip install peft==0.7.1
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.3
pip install huggingface-hub==0.19.4
```

