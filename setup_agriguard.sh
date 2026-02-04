#!/bin/bash

# 1. Create Directory Hierarchy
echo "Creating AgriGuard Directory Structure..."
mkdir -p data/raw/images
mkdir -p data/raw/financial
mkdir -p data/processed
mkdir -p src/models
mkdir -p src/utils
mkdir -p notebooks
mkdir -p app/pages
mkdir -p outputs/figures
mkdir -p weights

# 2. Create the Conda Environment File
echo "Generating environment.yml..."
cat <<EOL > environment.yml
name: agriguard_env
channels:
  - defaults
  - pytorch
  - conda-forge
dependencies:
  - python=3.9
  - pip
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - plotly
  - streamlit
  - opencv
  - tqdm
  # PyTorch (CPU version is usually default for lightweight envs, add cuda if you have GPU)
  - pytorch
  - torchvision
  - pip:
    - ultralytics  # For YOLOv8
    - roboflow     # For dataset management (optional but recommended)
    - pillow
EOL

# 3. Create a .gitignore for safety
echo "Creating .gitignore..."
cat <<EOL > .gitignore
# Data
data/
outputs/
weights/
*.pt
*.onnx

# Environment
.env
.ipynb_checkpoints/
__pycache__/
*.log
EOL

echo "=========================================="
echo "Setup Complete!"
echo "1. Run: conda env create -f environment.yml"
echo "2. Run: conda activate agriguard_env"
echo "=========================================="