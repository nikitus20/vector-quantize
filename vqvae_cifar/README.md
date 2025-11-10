# VQVAE CIFAR-10 with PyTorch Lightning

Clean, minimal implementation of VQVAE for CIFAR-10 using PyTorch Lightning and vector-quantize-pytorch.

## Files

- **`model.py`** (277 lines) - Encoder/Decoder architecture (DeepMind style)
- **`data.py`** (50 lines) - CIFAR-10 DataModule with augmentation
- **`vqvae.py`** (143 lines) - Lightning module with vector_quantize_pytorch integration
- **`test_vqvae.py`** - Quick test script

**Total: 470 lines** (clean, minimal, efficient)

## Key Features

✓ **Vector Quantization**: Uses `vector-quantize-pytorch` library
  - Codebook size: 512
  - Embedding dim: 64
  - EMA decay: 0.99
  - K-means initialization

✓ **Architecture**: DeepMind VQVAE
  - Encoder: Conv layers + Residual blocks
  - Decoder: Transposed convolutions
  - Hidden dim: 128

✓ **Training**: PyTorch Lightning
  - MSE reconstruction loss
  - VQ commitment loss
  - Perplexity tracking
  - Model checkpointing

## Usage

```bash
# Test that everything works
./venv/bin/python vqvae_cifar/test_vqvae.py

# Train (default: 512 codebook, 64 embedding dim, 128 hidden, auto-detect accelerator)
./venv/bin/python vqvae_cifar/vqvae.py

# Train with custom parameters
./venv/bin/python vqvae_cifar/vqvae.py \
    --num_embeddings 1024 \
    --embedding_dim 128 \
    --n_hid 256 \
    --batch_size 128 \
    --max_epochs 100

# With specific accelerator (auto-detects by default)
./venv/bin/python vqvae_cifar/vqvae.py --accelerator mps   # Mac M1/M2
./venv/bin/python vqvae_cifar/vqvae.py --accelerator gpu   # CUDA
./venv/bin/python vqvae_cifar/vqvae.py --accelerator cpu   # CPU only
```

## Experiment Tracking with Weights & Biases

Track experiments, visualize metrics, and compare runs using WandB:

```bash
# Enable WandB logging
./venv/bin/python vqvae_cifar/vqvae.py --use_wandb

# With custom project and run name
./venv/bin/python vqvae_cifar/vqvae.py \
    --use_wandb \
    --wandb_project "my-vqvae-experiments" \
    --wandb_name "baseline-run"

# With tags for organization
./venv/bin/python vqvae_cifar/vqvae.py \
    --use_wandb \
    --wandb_tags experiment baseline high_capacity

# Offline mode (sync later with `wandb sync`)
./venv/bin/python vqvae_cifar/vqvae.py \
    --use_wandb \
    --wandb_offline

# With WandB entity (team or username)
./venv/bin/python vqvae_cifar/vqvae.py \
    --use_wandb \
    --wandb_entity "my-team"
```

### What's Logged to WandB:

**Metrics:**
- Training & validation loss, PSNR, perplexity, codebook usage
- VQ-specific metrics (commitment loss, codebook loss)
- BA-VQ specific metrics (beta, entropy) if using BA mode

**Model Info:**
- Total parameters, trainable parameters, model size (MB)
- All hyperparameters and config

**Visualizations:**
- Sample image reconstructions every epoch (8 images)
- Model checkpoints (best model based on validation loss)

**Auto-generated Run Names:**
Format: `{vq_type}_nhid{n_hid}_cb{num_embeddings}_{timestamp}`
Example: `standard_nhid64_cb512_20250109_143022`

**Auto-tags:**
- VQ type (`standard` or `ba`)
- Architecture (`n_hid_64`)
- Accelerator (`mps`, `gpu`, `cpu`)
- Custom tags via `--wandb_tags`

## Arguments

### Model Architecture
- `--num_embeddings` (default: 512) - Codebook size
- `--embedding_dim` (default: 64) - Embedding dimension
- `--n_hid` (default: 64) - Hidden channels
- `--vq_type` (default: standard) - VQ type: 'standard' or 'ba'
- `--commitment_weight` (default: 0.25) - Commitment loss weight

### BA VectorQuantize (if --vq_type ba)
- `--beta_start` (default: 0.5) - Initial beta (inverse temperature)
- `--beta_end` (default: 3.0) - Final beta
- `--ba_iters` (default: 2) - BA iterations per forward pass
- `--entropy_weight` (default: 0.1) - Entropy regularization weight

### Training
- `--batch_size` (default: 128) - Batch size
- `--max_epochs` (default: 50) - Number of epochs
- `--data_dir` (default: ./data) - CIFAR-10 data directory
- `--num_workers` (default: 4) - DataLoader workers
- `--accelerator` (default: auto) - Accelerator to use (auto, cpu, gpu, mps)

### Experiment Tracking
- `--use_wandb` - Enable WandB logging
- `--wandb_project` (default: vqvae-cifar10) - WandB project name
- `--wandb_name` - Run name (auto-generated if not provided)
- `--wandb_entity` - WandB entity (username or team)
- `--wandb_tags` - Tags for the run (space-separated)
- `--wandb_offline` - Run in offline mode

## Expected Results

- **Reconstruction MSE**: ~0.04-0.05 after convergence
- **Perplexity**: ~400-500 (good codebook usage)
- **Cluster use**: Close to 512 (all codes used)

## Vector Quantize Integration

The code uses `vector-quantize-pytorch` library with:
```python
VectorQuantize(
    dim=64,                    # embedding_dim
    codebook_size=512,         # num_embeddings
    decay=0.99,                # EMA decay
    commitment_weight=0.25,    # commitment loss weight
    kmeans_init=True,          # k-means initialization
    kmeans_iters=10,
    accept_image_fmap=True     # accepts [B, C, H, W] format
)
```

Returns: `(quantized, indices, vq_loss)`
- `quantized`: Quantized embeddings
- `indices`: Codebook indices [B, 8, 8]
- `vq_loss`: Commitment loss (scalar)

## Code Structure

```
vqvae_cifar/
├── model.py          # Encoder, Decoder, ResidualBlocks
├── data.py           # CIFAR10Data Lightning DataModule
├── vqvae.py          # VQVAE Lightning Module + training script
├── test_vqvae.py     # Quick test
└── requirements.txt  # Dependencies
```

Clean and minimal - no unnecessary abstractions or complexity.
