# VQ-VAE Experiments - Lambda GPU Setup

## Quick Start

### 1. Setup on Lambda GPU Instance

```bash
# Clone repo
git clone <your-repo-url>
cd vector-quantize

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install vq_modules (local VQ implementations)
pip install -e .

# Install other dependencies
pip install -r vqvae_cifar/requirements.txt
pip install wandb

# Login to wandb (one-time)
wandb login
```

### 2. Run Experiments

```bash
# Make script executable
chmod +x run_experiments.py

# List available experiments
python run_experiments.py --list

# Run single experiment
python run_experiments.py --exp 11

# Run multiple experiments
python run_experiments.py --exp 11 12 13

# Run all new experiments (11, 12, 13)
python run_experiments.py --all
```

### 3. Monitor Results

Check WandB dashboard: https://wandb.ai/your-username/vqvae-cifar10

## New Experiments (Zero Entropy)

All new experiments use **entropy_weight=0.0** for fair comparison with standard VQ.

### Experiment 11: Zero-Start Standard-End
- β: 0.0 → 3.0 (maximum exploration → standard exploitation)
- ba_iters: 2
- **Goal**: Maximize perplexity spike, early transition

### Experiment 12: Ultra-Low Hard-End
- β: 0.05 → 4.0 (near-max exploration → aggressive exploitation)
- ba_iters: 2
- **Goal**: Best final PSNR with large spike

### Experiment 13: Low-Start Standard-End 1-Iter
- β: 0.1 → 3.0 (like exp3 → standard)
- ba_iters: 1 (faster!)
- **Goal**: Efficient, balanced performance

## Expected Behavior

All three should:
- ✅ Large perplexity spike (~150-200+) around step 2000
- ✅ Final PSNR ≥ 22.16 dB (matching or exceeding exp8)
- ✅ No entropy loss (same objective as standard VQ)

## File Structure

```
vector-quantize/
├── vqvae_cifar/
│   ├── vqvae.py           # Main training script
│   ├── model.py           # Model architecture
│   ├── data.py            # CIFAR-10 data loader
│   └── requirements.txt   # Dependencies
├── vector_quantize_pytorch/
│   ├── vector_quantize_pytorch.py  # Standard VQ
│   └── ba_vector_quantize.py       # BA VQ
├── run_experiments.py     # Experiment runner (USE THIS)
├── EXPERIMENT_SUMMARY.md  # Previous results analysis
└── QUICK_REFERENCE.md     # Quick lookup guide
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in vqvae.py:
```bash
python vqvae_cifar/vqvae.py --batch_size 64 ...
```

### Slow Training
Check GPU utilization:
```bash
nvidia-smi
```

Should show high GPU usage during training.

### WandB Issues
If wandb sync fails, run offline:
```bash
python run_experiments.py --exp 11  # (will still log locally)
```

## Next Steps After Experiments

1. Check WandB for results
2. Compare perplexity spikes at step ~2000
3. Compare final PSNR to baseline (22.34 dB) and exp8 (22.16 dB)
4. Analyze which β schedule works best

## Reference: Previous Best Results

- **Standard VQ**: 22.34 dB, 283.4 perplexity, 96.3% usage
- **exp8 (BA low entropy)**: 22.16 dB, 135.3 perplexity, 47.4% usage
- **exp3 (BA exploratory)**: 21.95 dB, 100.2 perplexity, 39% usage (huge spike!)

Goal: Match or beat exp8's PSNR (22.16 dB) while getting exp3's spike dynamics.
