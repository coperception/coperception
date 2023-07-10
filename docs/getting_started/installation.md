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

Create and activate a conda environment named coperception
```bash
conda create -n coperception python=3.7
conda activate coperception
```

Install dependencies via pip:  
(Make sure your Python version is `3.7`)
```bash
pip install -r requirements.txt
```

### 2. CoPerception library
Use pip to install `coperception` library:
```bash
pip install -e .
```
This installs and links `coperception` library to code in `./coperception` directory.

### 3. CUDA support
Coperception requires CUDA to run on the GPU.  
Go to PyTorch's [installation documentation](https://pytorch.org/get-started/locally/) to set up CUDA and PyTorch.