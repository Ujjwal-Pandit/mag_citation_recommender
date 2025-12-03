# Environment Management Commands

## Conda Environment

### Delete existing environment
```bash
# First deactivate if currently active
conda deactivate

# Remove the environment
conda env remove -n mag_citation
```

### Create new environment
```bash
# Create new conda environment
conda create -n mag_citation python=3.9 -y

# Activate it
conda activate mag_citation
```

### List all environments
```bash
conda env list
```

### Export environment (save current state)
```bash
conda env export > environment.yml
```

### Recreate from exported file
```bash
conda env create -f environment.yml
```

---

## Virtual Environment (venv)

### Delete existing environment
```bash
# First deactivate if currently active
deactivate

# Remove the directory
rm -rf venv
```

### Create new environment
```bash
# Create new virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

---

## Quick Reset (Conda)

Complete reset of the `mag_citation` environment:
```bash
conda deactivate
conda env remove -n mag_citation
conda create -n mag_citation python=3.9 -y
conda activate mag_citation
```

