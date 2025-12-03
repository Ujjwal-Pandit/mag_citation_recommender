# Running on GPU Cluster

## Quick Setup Commands

After SSHing into the GPU cluster, run these commands:

### 1. Clone the repository
```bash
git clone https://github.com/Ujjwal-Pandit/mag_citation_recommender.git
cd mag_citation_recommender
```

### 2. Set up Python environment (choose one method)

#### Option A: Using conda (recommended)
```bash
# Create a new conda environment
conda create -n mag_citation python=3.9 -y
conda activate mag_citation

# Install PyTorch with CUDA support (auto-detects CUDA version)
# Try CUDA 11.8 first (most common)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y || \
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install torch-geometric sentence-transformers tqdm numpy
```

#### Option A2: Delete and recreate environment (if needed)
```bash
# Delete existing environment
conda deactivate  # if currently activated
conda env remove -n mag_citation

# Create fresh environment
conda create -n mag_citation python=3.9 -y
conda activate mag_citation
```

#### Option B: Using venv
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (auto-detects)
# Try CUDA 11.8 first, then 12.1 if that fails
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 || \
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install torch-geometric sentence-transformers tqdm numpy
```

#### Option B2: Using auto-detection script (easiest)
```bash
# Create environment first
python3 -m venv venv
source venv/bin/activate

# Run the auto-detection installer
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### 3. Request GPU resources (if using SLURM)
```bash
# Request an interactive GPU session
srun --gres=gpu:1 --time=4:00:00 --pty bash

# Or submit as a job (create a job script)
```

### 4. Run the notebook

#### Option A: Convert to Python script and run
```bash
# Convert notebook to Python script
jupyter nbconvert --to script notebooks_01_continued.ipynb

# Run the script
python notebooks_01_continued.py
```

#### Option B: Run with Jupyter (if available on cluster)
```bash
# Start Jupyter on a compute node
jupyter notebook --no-browser --port=8888

# Then SSH tunnel from your local machine:
# ssh -L 8888:localhost:8888 username@cluster-address
```

#### Option C: Use papermill to execute notebook
```bash
pip install papermill
papermill notebooks_01_continued.ipynb output.ipynb
```

### 5. Verify GPU availability
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Complete One-Liner Setup (if conda is available)
```bash
git clone https://github.com/Ujjwal-Pandit/mag_citation_recommender.git && \
cd mag_citation_recommender && \
conda create -n mag_citation python=3.9 -y && \
conda activate mag_citation && \
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
pip install torch-geometric sentence-transformers tqdm numpy && \
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## SLURM Job Script Example

Create a file `run_notebook.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=mag_citation
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Activate conda environment
source ~/.bashrc
conda activate mag_citation

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/mag_citation_recommender

# Convert and run notebook
jupyter nbconvert --to script notebooks_01_continued.ipynb
python notebooks_01_continued.py
```

Then submit with:
```bash
sbatch run_notebook.sh
```

## Notes
- **No CUDA version checking needed**: Use `./install_dependencies.sh` script for auto-detection
- If you can't check CUDA version with `nvidia-smi`, the script will try different versions automatically
- After installation, verify GPU: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Monitor GPU usage: `watch -n 1 nvidia-smi` (if allowed by admin)
- The notebook will automatically download data files if they don't exist
- Checkpoints will be saved in `checkpoints/` directory

