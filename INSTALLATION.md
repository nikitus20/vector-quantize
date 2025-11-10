# vq_modules Installation Guide

## Quick Install

```bash
# Clone the repository
git clone git@github.com:nikitus20/vector-quantize.git
cd vector-quantize

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install vq_modules in editable mode
pip install -e .

# Install other dependencies
pip install -r vqvae_cifar/requirements.txt
pip install wandb

# Login to wandb
wandb login
```

## What `pip install -e .` Does

The `-e` flag installs the package in **editable mode**:
- Makes `vq_modules` importable from anywhere
- Changes to the code are immediately reflected (no reinstall needed)
- Perfect for development and experimentation

## Verification

Test that installation worked:

```bash
# Test from any directory
python -c "from vq_modules import VectorQuantize, BAVectorQuantize; print('✓ Installation successful!')"

# Test from vqvae_cifar directory
cd vqvae_cifar
python -c "from vq_modules import VectorQuantize, BAVectorQuantize; print('✓ Works from subdirectory!')"
```

## What's Included in vq_modules

- `VectorQuantize`: Standard vector quantization (from lucidrains' library)
- `BAVectorQuantize`: Blahut-Arimoto inspired vector quantization (our implementation)

## Package Structure

```
vector-quantize/
├── vq_modules/                    # Local VQ package
│   ├── __init__.py                # Exports VectorQuantize, BAVectorQuantize
│   ├── vector_quantize_pytorch.py # Standard VQ implementation
│   ├── ba_vector_quantize.py      # BA VQ implementation
│   └── ...                        # Other VQ variants
├── setup.py                       # pip install configuration
├── pyproject.toml                 # Modern Python packaging config
└── vqvae_cifar/                   # Training code
    └── vqvae.py                   # Imports: from vq_modules import ...
```

## Troubleshooting

### ModuleNotFoundError: No module named 'vq_modules'

**Solution**: Install the package in editable mode:
```bash
pip install -e .
```

### Import works in root but not in vqvae_cifar/

This was the original issue! **Solution**: `pip install -e .` makes it work from anywhere.

### Want to uninstall?

```bash
pip uninstall vq_modules
```

## For Lambda GPU

Follow the Quick Install steps above. The editable install ensures imports work regardless of where you run the experiments from.

## Technical Details

### Why editable install?

Without `-e`:
- Package is copied to site-packages
- Must reinstall after every code change
- Not ideal for development

With `-e`:
- Package links to source directory
- Changes immediately available
- Perfect for experimentation

### Package metadata

Configured in `pyproject.toml`:
- Package name: `vq_modules`
- Version: `0.1.0`
- Dependencies: `torch>=2.0`, `einops>=0.6.0`
- Build system: `hatchling`

### Import resolution

After `pip install -e .`:
```python
from vq_modules import VectorQuantize, BAVectorQuantize
```

Python finds it via:
1. Checks site-packages for `vq_modules`
2. Finds link to `/path/to/vector-quantize/vq_modules/`
3. Imports from source directory

No matter which directory you're in! ✓
