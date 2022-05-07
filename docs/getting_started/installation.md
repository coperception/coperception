# Installation

## Supported environments
- **System**: Ubuntu 18.4; RHEL 8.4
- **Python**: 3.7
---
## Installation
### 1. Dependencies
Download the GitHub repository:
```bash
git clone https://github.com/coperception/coperception.git
cd coperception
```

Create a conda environment with the dependencies:
```bash
conda env create -f environment.yml
conda activate coperception
```

If conda installation failed, install the dependencies through pip:  
(Make sure your Python version is `3.7`)
```bash
pip install -r requirements.txt
```

### 2. CUDA support
Coperception requires CUDA to run on the GPU.  
Go to PyTorch's [installation documentation](https://pytorch.org/get-started/locally/) to set up CUDA and PyTorch.