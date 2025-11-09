"""
Blahut-Arimoto Vector Quantization

Implements a rate-distortion optimized vector quantizer using the Blahut-Arimoto algorithm.
Uses soft assignments with temperature annealing and iterative E-M updates.
"""

from math import pi, cos
from typing import Optional, Tuple, Union
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast

from einops import rearrange, reduce, pack, unpack

from vector_quantize_pytorch.vector_quantize_pytorch import kmeans

# Named tuple for clean return values
Return = namedtuple('Return', ['quantized', 'indices', 'loss'])
AuxDict = namedtuple('AuxDict', [
    'loss',
    'commitment_loss',
    'codebook_loss',
    'entropy_loss',
    'beta',
    'H_Q',
    'perplexity',
    'usage'
])

# Helper functions

def exists(val):
    """Check if value is not None"""
    return val is not None

def default(val, d):
    """Return val if exists, otherwise d"""
    return val if exists(val) else d

def identity(t):
    """Identity function"""
    return t

def safe_div(num, den, eps=1e-6):
    """Safe division with epsilon"""
    return num / den.clamp(min=eps)

def log(t, eps=1e-20):
    """Safe log with epsilon"""
    return torch.log(t.clamp(min=eps))

def entropy(prob, dim=-1, eps=1e-20):
    """Compute entropy: -sum(p * log(p))"""
    return -torch.sum(prob * log(prob, eps=eps), dim=dim)

def pack_one(t, pattern):
    """Pack tensor with pattern, return unpacker function"""
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern=None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one

def l2norm(t, dim=-1, eps=1e-6):
    """L2 normalization"""
    return F.normalize(t, p=2, dim=dim, eps=eps)

class BAVectorQuantize(nn.Module):
    """
    Blahut-Arimoto Vector Quantization

    Implements rate-distortion optimized VQ using the Blahut-Arimoto algorithm.
    Features soft assignments with temperature (beta) annealing and iterative
    expectation-maximization updates.

    Args:
        dim: Input feature dimension
        codebook_size: Number of codebook vectors
        use_cosine_sim: Use cosine similarity instead of Euclidean distance
        commitment_weight: Weight for commitment loss
        codebook_weight: Weight for codebook loss
        entropy_weight: Weight for entropy regularization on Q
        beta_start: Initial beta (inverse temperature) for soft assignments
        beta_end: Final beta value after annealing
        ba_iters: Number of BA fixed-point iterations per forward pass
        ema_pi_decay: EMA decay for prior pi (dataset-level marginal)
        chunk_size: Chunk size for memory-efficient distance computation
        accept_image_fmap: Accept image feature maps [B,C,H,W]
        channel_last: Expect channels last format
        straight_through: Use straight-through estimator (vs other gradient tricks)
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        use_cosine_sim: bool = False,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
        entropy_weight: float = 0.01,
        beta_start: float = 0.5,
        beta_end: float = 3.0,
        ba_iters: int = 2,
        ema_pi_decay: float = 0.99,
        chunk_size: int = 8192,
        accept_image_fmap: bool = False,
        channel_last: bool = True,
        straight_through: bool = True,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
    ):
        super().__init__()

        # Store configuration
        self.dim = dim
        self.codebook_size = codebook_size
        self.use_cosine_sim = use_cosine_sim
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.entropy_weight = entropy_weight
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.ba_iters = ba_iters
        self.ema_pi_decay = ema_pi_decay
        self.chunk_size = chunk_size
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        self.straight_through = straight_through
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        # Initialize codebook
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.xavier_uniform_(self.codebook.weight)

        # Initialize beta (will be updated during training)
        self.register_buffer('beta', torch.tensor(beta_start))

        # Initialize prior pi (uniform initially, updated via EMA)
        init_pi = torch.ones(codebook_size) / codebook_size
        self.register_buffer('pi', init_pi)

        # Zero tensor for loss initialization
        self.register_buffer('zero', torch.tensor(0.))

        # Track if kmeans initialization has been done
        self.register_buffer('initted', torch.tensor(False))

    def _normalize_codebook(self):
        """Normalize codebook vectors (for cosine similarity mode)"""
        if self.use_cosine_sim:
            with torch.no_grad():
                self.codebook.weight.data = l2norm(self.codebook.weight.data)

    def _update_beta(self, step: Optional[int], total_steps: Optional[int]):
        """Update beta using cosine annealing schedule"""
        if step is None or total_steps is None or total_steps <= 1:
            return

        t = min(step / (total_steps - 1), 1.0)
        new_beta = self.beta_start + (self.beta_end - self.beta_start) * 0.5 * (1 - cos(pi * t))
        self.beta.data = torch.tensor(new_beta, device=self.beta.device)

    def _compute_distances_chunked(self, z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between inputs and codebook vectors in chunks.

        Args:
            z: Input tensor [N, D]
            codebook: Codebook tensor [K, D]

        Returns:
            distances: [N, K] distance matrix
        """
        N = z.shape[0]
        K = codebook.shape[0]
        device = z.device

        # Prepare codebook based on distance metric
        if self.use_cosine_sim:
            # Normalize both z and codebook
            z = l2norm(z)
            codebook = l2norm(codebook)

        # Compute distances in chunks to avoid OOM
        distances = torch.zeros(N, K, device=device)

        for start_idx in range(0, N, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, N)
            z_chunk = z[start_idx:end_idx]  # [chunk_size, D]

            if self.use_cosine_sim:
                # Cosine distance: d = 1 - cos(z, e)
                # cos(z, e) = z · e (both normalized)
                cos_sim = einsum('n d, k d -> n k', z_chunk, codebook)
                dist_chunk = 1.0 - cos_sim
            else:
                # Euclidean distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
                z_sq = torch.sum(z_chunk ** 2, dim=-1, keepdim=True)  # [chunk_size, 1]
                e_sq = torch.sum(codebook ** 2, dim=-1, keepdim=False)  # [K]
                ze = einsum('n d, k d -> n k', z_chunk, codebook)  # [chunk_size, K]
                dist_chunk = z_sq + e_sq - 2 * ze
                dist_chunk = dist_chunk.clamp(min=1e-8)  # Numerical stability

            distances[start_idx:end_idx] = dist_chunk

        return distances

    def _ba_iterations(self, distances: torch.Tensor, Q_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Blahut-Arimoto E-step iterations.

        Args:
            distances: [N, K] distance matrix
            Q_init: [K] initial Q distribution (prior)

        Returns:
            P: [N, K] soft assignment probabilities
            Q: [K] updated marginal distribution
        """
        N, K = distances.shape
        device = distances.device

        Q = Q_init.clone()

        for _ in range(self.ba_iters):
            # E-step: P(k|z) ∝ Q(k) * exp(-beta * d(z, e_k))
            # logits = log(Q) - beta * d
            log_Q = log(Q)  # [K]
            logits = log_Q.unsqueeze(0) - self.beta * distances  # [N, K]

            # Numerical stability: subtract max before softmax (log-sum-exp trick)
            logits_max = logits.max(dim=-1, keepdim=True)[0]
            logits_stable = logits - logits_max

            # Softmax to get probabilities
            P = F.softmax(logits_stable, dim=-1)  # [N, K]

            # M-step: Update Q as marginal over batch
            # Q(k) = (1/N) * sum_n P(k|z_n)
            Q = P.mean(dim=0)  # [K]

        return P, Q

    def _update_prior(self, Q: torch.Tensor):
        """Update EMA prior pi"""
        if self.training:
            with torch.no_grad():
                self.pi.data = self.ema_pi_decay * self.pi.data + (1 - self.ema_pi_decay) * Q

    def _init_with_kmeans(self, x: torch.Tensor):
        """Initialize codebook using k-means clustering on first batch"""
        if self.initted or not self.kmeans_init:
            return

        # Flatten x to [N, D]
        if x.ndim == 4 and self.accept_image_fmap:
            x_flat = rearrange(x, 'b c h w -> (b h w) c')
        elif x.ndim == 3:
            x_flat = rearrange(x, 'b n d -> (b n) d')
        else:
            x_flat = x

        # Sample data if too large
        max_samples = 100000
        if x_flat.shape[0] > max_samples:
            indices = torch.randperm(x_flat.shape[0])[:max_samples]
            x_sample = x_flat[indices]
        else:
            x_sample = x_flat

        # Add batch dimension for kmeans function: [N, D] -> [1, N, D]
        x_sample = x_sample.unsqueeze(0)

        # Run k-means (returns tuple of (embed, cluster_size))
        codebook_init, _ = kmeans(
            x_sample,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim=self.use_cosine_sim,
        )

        # Remove batch dimension: [1, K, D] -> [K, D]
        codebook_init = codebook_init.squeeze(0)

        # Update codebook
        self.codebook.weight.data.copy_(codebook_init)
        self.initted.data.copy_(torch.tensor(True))

    @autocast('cuda', enabled = False)
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Union[Return, Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, dict]]]:
        """
        Forward pass of BA vector quantization.

        Args:
            x: Input tensor
                - Image fmaps: [B, C, H, W] if accept_image_fmap=True
                - Flat: [B, N, D]
            return_all: Return auxiliary metrics dict instead of scalar loss
            step: Current training step (for beta annealing)
            total_steps: Total training steps (for beta annealing)

        Returns:
            If return_all=False:
                (quantized, indices, loss)
            If return_all=True:
                (quantized, indices, aux_dict)
        """
        # Force float32 for numerical stability
        x = x.float()

        # Initialize codebook with k-means on first batch
        if self.training:
            self._init_with_kmeans(x)

        # Update beta based on training schedule
        if self.training:
            self._update_beta(step, total_steps)

        # Normalize codebook if using cosine similarity
        if self.use_cosine_sim and self.training:
            self._normalize_codebook()

        # Store original shape for later reconstruction
        orig_shape = x.shape
        device = x.device

        # Flatten input to [N, D] format
        if self.accept_image_fmap and x.ndim == 4:
            # Image: [B, C, H, W] -> [B*H*W, C]
            batch_size, channels, height, width = x.shape
            x = rearrange(x, 'b c h w -> (b h w) c')
            need_image_reshape = True
        elif x.ndim == 3:
            # Sequence: [B, N, D] -> [B*N, D]
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            x = rearrange(x, 'b n d -> (b n) d')
            need_image_reshape = False
        else:
            # Already flat
            need_image_reshape = False

        z = x
        N, D = z.shape

        # Get codebook
        codebook = self.codebook.weight  # [K, D]

        # Compute distances [N, K]
        distances = self._compute_distances_chunked(z, codebook)

        # Run BA iterations
        P, Q = self._ba_iterations(distances, self.pi)

        # Update prior pi with EMA
        self._update_prior(Q)

        # Soft quantization: z_q_soft = sum_k P(k|z) * e_k
        z_q_soft = einsum('n k, k d -> n d', P, codebook)  # [N, D]

        # Hard quantization: take argmax
        indices = P.argmax(dim=-1)  # [N]
        z_q_hard = F.embedding(indices, codebook)  # [N, D]

        # Straight-through estimator
        if self.straight_through:
            z_q = z_q_soft + (z_q_hard - z_q_soft).detach()
        else:
            # Could add rotation trick or other gradient flow methods here
            z_q = z_q_soft + (z_q_hard - z_q_soft).detach()

        # Compute losses
        loss = self.zero.clone()

        # Commitment loss: ||z - sg(z_q_soft)||^2
        commitment_loss = F.mse_loss(z, z_q_soft.detach())
        loss = loss + self.commitment_weight * commitment_loss

        # Codebook loss: ||z_q_soft - sg(z)||^2
        codebook_loss = F.mse_loss(z_q_soft, z.detach())
        loss = loss + self.codebook_weight * codebook_loss

        # Entropy loss: KL(Q || uniform) or -H(Q)
        # We use -H(Q) to encourage uniform usage
        H_Q = entropy(Q)
        entropy_loss = -H_Q
        loss = loss + self.entropy_weight * entropy_loss

        # Compute metrics
        perplexity = torch.exp(H_Q)
        usage = (Q > 1e-6).float().mean()  # Fraction of codes with non-negligible usage

        # Reshape back to original format
        if self.accept_image_fmap and len(orig_shape) == 4:
            # Image: [B*H*W, C] -> [B, C, H, W]
            z_q = rearrange(z_q, '(b h w) c -> b c h w', b=batch_size, h=height, w=width)
            indices_out = rearrange(indices, '(b h w) -> b h w', b=batch_size, h=height, w=width)
        elif len(orig_shape) == 3:
            # Sequence: [B*N, D] -> [B, N, D]
            z_q = rearrange(z_q, '(b n) d -> b n d', b=batch_size, n=seq_len)
            indices_out = rearrange(indices, '(b n) -> b n', b=batch_size, n=seq_len)
        else:
            # Keep as is
            indices_out = indices

        # Return values
        if return_all:
            aux_dict = {
                'loss': loss.detach(),
                'commitment_loss': commitment_loss.detach(),
                'codebook_loss': codebook_loss.detach(),
                'entropy_loss': entropy_loss.detach(),
                'beta': self.beta.detach(),
                'H_Q': H_Q.detach(),
                'perplexity': perplexity.detach(),
                'usage': usage.detach(),
            }
            return z_q, indices_out, aux_dict
        else:
            return z_q, indices_out, loss
