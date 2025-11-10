"""
VQVAE training with PyTorch Lightning
"""

from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
import torchvision
import wandb

from vq_modules import VectorQuantize, BAVectorQuantize
from data import CIFAR10Data
from model import DeepMindEncoder, DeepMindDecoder


class ImageReconstructionLogger(Callback):
    """Callback to log image reconstructions to WandB"""

    def __init__(self, num_samples=8):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log sample reconstructions at the end of validation"""
        if not isinstance(trainer.logger, WandbLogger):
            return

        # Get a batch from validation set
        val_dataloader = trainer.val_dataloaders
        batch = next(iter(val_dataloader))
        images, _ = batch

        # Limit to num_samples
        images = images[:self.num_samples].to(pl_module.device)

        # Generate reconstructions
        with torch.no_grad():
            pl_module.eval()
            reconstructions, _, _, _ = pl_module(images, return_vq_metrics=False)
            pl_module.train()

        # Create comparison grid
        # Interleave original and reconstruction
        comparison = torch.stack([images, reconstructions], dim=1).flatten(0, 1)
        grid = torchvision.utils.make_grid(
            comparison,
            nrow=2,  # 2 columns: original, reconstruction
            normalize=True,
            value_range=(0, 1)
        )

        # Log to wandb
        trainer.logger.experiment.log({
            "reconstructions": wandb.Image(
                grid,
                caption=f"Epoch {trainer.current_epoch}"
            ),
            "global_step": trainer.global_step,
        })


class VQVAE(pl.LightningModule):

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        n_hid: int = 64,
        vq_type: str = 'standard',
        commitment_weight: float = 0.25,
        # BA VQ specific parameters
        beta_start: float = 0.5,
        beta_end: float = 3.0,
        ba_iters: int = 2,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.encoder = DeepMindEncoder(
            input_channels=3,
            n_hid=n_hid
        )

        # Projection from encoder output to embedding_dim
        # encoder outputs 2*n_hid channels, VQ expects embedding_dim
        self.pre_quant_conv = torch.nn.Conv2d(
            self.encoder.output_channels,
            embedding_dim,
            kernel_size=1
        )

        # Vector Quantizer - factory based on type
        self.vq = self._create_vq_layer(
            vq_type=vq_type,
            dim=embedding_dim,
            codebook_size=num_embeddings,
            commitment_weight=commitment_weight,
            beta_start=beta_start,
            beta_end=beta_end,
            ba_iters=ba_iters,
            entropy_weight=entropy_weight,
        )

        # Projection from embedding_dim to decoder input
        self.post_quant_conv = torch.nn.Conv2d(
            embedding_dim,
            embedding_dim,
            kernel_size=1
        )

        # Decoder
        self.decoder = DeepMindDecoder(
            n_init=embedding_dim,
            n_hid=n_hid,
            output_channels=3
        )

    def _create_vq_layer(
        self,
        vq_type: str,
        dim: int,
        codebook_size: int,
        commitment_weight: float,
        beta_start: float,
        beta_end: float,
        ba_iters: int,
        entropy_weight: float,
    ):
        """Factory function to create VQ layer based on type."""
        if vq_type == 'standard':
            return VectorQuantize(
                dim=dim,
                codebook_size=codebook_size,
                decay=0.99,
                commitment_weight=commitment_weight,
                kmeans_init=True,
                kmeans_iters=10,
                accept_image_fmap=True
            )
        elif vq_type == 'ba':
            return BAVectorQuantize(
                dim=dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight,
                codebook_weight=1.0,
                entropy_weight=entropy_weight,
                beta_start=beta_start,
                beta_end=beta_end,
                ba_iters=ba_iters,
                accept_image_fmap=True,
                kmeans_init=True,
                kmeans_iters=10,
            )
        else:
            raise ValueError(f"Unknown vq_type: {vq_type}. Choose 'standard' or 'ba'.")

    def forward(self, x, return_vq_metrics=False):
        # Encoder: [B, 3, 32, 32] -> [B, 2*n_hid, 8, 8]
        z = self.encoder(x)

        # Pre-quantization projection: [B, 2*n_hid, 8, 8] -> [B, embedding_dim, 8, 8]
        z = self.pre_quant_conv(z)

        # Handle BA VQ with step tracking
        if self.hparams.vq_type == 'ba' and return_vq_metrics:
            # Calculate total steps from trainer
            # trainer.max_steps is -1 when using max_epochs, so calculate manually
            if self.trainer and hasattr(self.trainer, 'num_training_batches'):
                total_steps = self.trainer.max_epochs * self.trainer.num_training_batches
            else:
                total_steps = None

            z_q, indices, aux_dict = self.vq(
                z,
                return_all=True,
                step=self.global_step,
                total_steps=total_steps
            )
            vq_loss = aux_dict['loss']
            vq_metrics = aux_dict
        else:
            # VQ: [B, embedding_dim, 8, 8] -> [B, embedding_dim, 8, 8], indices
            z_q, indices, vq_loss = self.vq(z)
            vq_metrics = None

        # Post-quantization projection: [B, embedding_dim, 8, 8] -> [B, embedding_dim, 8, 8]
        z_q = self.post_quant_conv(z_q)

        # Decoder: [B, embedding_dim, 8, 8] -> [B, 3, 32, 32]
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices, vq_metrics

    def _compute_metrics(self, indices):
        """Compute perplexity and codebook usage rate."""
        encodings = F.one_hot(indices.flatten(), self.hparams.num_embeddings).float()
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        usage = (avg_probs > 0).float().mean()
        return perplexity, usage

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, vq_loss, indices, vq_metrics = self(x, return_vq_metrics=True)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        # PSNR (images in [0,1] range)
        psnr = -10 * torch.log10(recon_loss)

        # Perplexity and usage rate
        perplexity, usage = self._compute_metrics(indices)

        # Total loss
        loss = recon_loss + vq_loss

        # Standard metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon', recon_loss)
        self.log('train_psnr', psnr, prog_bar=True)
        self.log('train_perplexity', perplexity)
        self.log('train_usage', usage)
        self.log('train_vq', vq_loss)

        # BA VQ specific metrics
        if vq_metrics is not None:
            self.log('train_ba_beta', vq_metrics.get('beta', 0))
            self.log('train_ba_entropy', vq_metrics.get('H_Q', 0))
            self.log('train_ba_commitment', vq_metrics.get('commitment_loss', 0))
            self.log('train_ba_codebook', vq_metrics.get('codebook_loss', 0))
            self.log('train_ba_entropy_loss', vq_metrics.get('entropy_loss', 0))

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, vq_loss, indices, vq_metrics = self(x, return_vq_metrics=True)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        # PSNR
        psnr = -10 * torch.log10(recon_loss)

        # Perplexity and usage rate
        perplexity, usage = self._compute_metrics(indices)

        # Standard metrics
        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_psnr', psnr, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)
        self.log('val_usage', usage, prog_bar=True)
        self.log('val_vq_loss', vq_loss)

        # BA VQ specific metrics
        if vq_metrics is not None:
            self.log('val_ba_beta', vq_metrics.get('beta', 0))
            self.log('val_ba_entropy', vq_metrics.get('H_Q', 0))
            self.log('val_ba_commitment', vq_metrics.get('commitment_loss', 0))
            self.log('val_ba_codebook', vq_metrics.get('codebook_loss', 0))
            self.log('val_ba_entropy_loss', vq_metrics.get('entropy_loss', 0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        return optimizer


def cli_main():
    pl.seed_everything(1337)

    # Arguments
    parser = ArgumentParser()
    # Model architecture
    parser.add_argument("--num_embeddings", type=int, default=512,
                        help='Codebook size')
    parser.add_argument("--embedding_dim", type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument("--n_hid", type=int, default=64,
                        help='Number of hidden units')

    # VQ type and shared parameters
    parser.add_argument("--vq_type", type=str, default='standard',
                        choices=['standard', 'ba'],
                        help='VQ type: standard or ba (Blahut-Arimoto)')
    parser.add_argument("--commitment_weight", type=float, default=0.25,
                        help='Commitment loss weight (used by both VQ types)')

    # BA VQ specific parameters
    parser.add_argument("--beta_start", type=float, default=0.5,
                        help='BA VQ: Initial beta (inverse temperature)')
    parser.add_argument("--beta_end", type=float, default=3.0,
                        help='BA VQ: Final beta (inverse temperature)')
    parser.add_argument("--ba_iters", type=int, default=2,
                        help='BA VQ: Number of BA iterations per forward pass')
    parser.add_argument("--entropy_weight", type=float, default=0.1,
                        help='BA VQ: Entropy regularization weight (increased from 0.01)')

    # Data and training
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accelerator", type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu', 'mps'],
                        help='Accelerator to use (auto, cpu, gpu, mps)')

    # WandB logging
    parser.add_argument("--wandb_project", type=str, default='vqvae-cifar10',
                        help='WandB project name')
    parser.add_argument("--wandb_name", type=str, default=None,
                        help='WandB run name (auto-generated if not provided)')
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help='WandB entity (username or team)')
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=None,
                        help='Tags for the run (e.g., --wandb_tags experiment baseline)')
    parser.add_argument("--use_wandb", action='store_true',
                        help='Enable WandB logging')
    parser.add_argument("--wandb_offline", action='store_true',
                        help='Run WandB in offline mode')

    args = parser.parse_args()

    # Auto-detect accelerator if set to 'auto'
    if args.accelerator == 'auto':
        if torch.cuda.is_available():
            accelerator = 'gpu'
        elif torch.backends.mps.is_available():
            accelerator = 'mps'
        else:
            accelerator = 'cpu'
    else:
        accelerator = args.accelerator

    # Data and Model
    data = CIFAR10Data(args)
    model = VQVAE(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        n_hid=args.n_hid,
        vq_type=args.vq_type,
        commitment_weight=args.commitment_weight,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        ba_iters=args.ba_iters,
        entropy_weight=args.entropy_weight,
    )

    # WandB Logger (optional)
    logger = None
    if args.use_wandb:
        import os
        from datetime import datetime

        # Auto-generate run name if not provided
        if args.wandb_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb_name = f"{args.vq_type}_nhid{args.n_hid}_cb{args.num_embeddings}_{timestamp}"
        else:
            wandb_name = args.wandb_name

        # Prepare tags
        tags = args.wandb_tags if args.wandb_tags else []
        tags.append(args.vq_type)
        tags.append(f"n_hid_{args.n_hid}")
        tags.append(accelerator)

        # Set offline mode if requested
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_name,
            tags=tags,
            config=vars(args),
            log_model=True,  # Log model checkpoints to wandb
        )

        # Log additional model info
        logger.experiment.config.update({
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() for p in model.parameters()) * 4 / (1024**2),  # Assuming float32
        })

    # Trainer
    callbacks = [ModelCheckpoint(monitor='val_recon_loss', mode='min', save_top_k=1)]

    # Add image logging callback if using wandb
    if args.use_wandb:
        callbacks.append(ImageReconstructionLogger(num_samples=8))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=1,
        logger=logger,
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    cli_main()
