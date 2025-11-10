"""
Blahut-Arimoto Inspired Vector Quantization

Implements a rate-distortion optimized vector quantizer inspired by the Blahut-Arimoto algorithm.
This implementation follows the theoretical approach described in the BA-inspired VQ paper, featuring:

1. Soft assignments via p(k|e) = softmax(log p(k) - beta * ||e - c_k||^2)
2. BA-style alternating optimization (E-step for assignments, M-step for codes)
3. Temperature annealing (beta) from low to high for exploration -> exploitation
4. Weighted EMA codebook updates following BA's reproduction update
5. Fully differentiable forward pass (no STE needed for soft path)

Theory: Achieves rate-distortion curve lower bound by minimizing I(X; Z) + beta * E[d(X, Z)]
"""

from math import pi, cos
from typing import Optional, Tuple, Union
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast

from einops import rearrange, reduce, pack, unpack

from vq_modules.vector_quantize_pytorch import kmeans

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
    Blahut-Arimoto Inspired Vector Quantization

    Implements rate-distortion optimized VQ using soft assignments and BA-inspired updates.
    
    Key Theoretical Components:
    - Soft assignments: p(k|e) = softmax(log p(k) - beta * d(e, c_k))
    - BA alternating optimization: Update assignments (E-step), then codes (M-step)
    - Temperature annealing: Start soft (explore), gradually harden (exploit)
    - Weighted centroids: c_k = sum_i p(k|e_i) * e_i / sum_i p(k|e_i)
    
    Args:
        dim: Input feature dimension
        codebook_size: Number of codebook vectors (K)
        use_cosine_sim: Use cosine similarity instead of Euclidean distance
        commitment_weight: Weight for commitment loss (pulls encoder to codes)
        codebook_weight: Weight for codebook loss (pulls codes to encoder)
        entropy_weight: Weight for entropy regularization on Q (encourages uniform usage)
        beta_start: Initial beta (inverse temperature) - low for soft exploration
        beta_end: Final beta after annealing - high for hard assignment
        num_cycles: Number of beta annealing cycles (default 1, set >1 for cyclical)
        ba_iters: Number of BA E-M iterations per forward pass
        ema_pi_decay: EMA decay for prior p(k) (dataset-level marginal)
        ema_code_decay: EMA decay for codebook updates (BA-inspired weighted EMA)
        chunk_size: Chunk size for memory-efficient distance computation
        accept_image_fmap: Accept image feature maps [B,C,H,W]
        channel_last: Expect channels last format
        use_soft_only: Use only soft quantization (no STE, fully differentiable)
        kmeans_init: Initialize codebook with k-means on first batch
        kmeans_iters: Number of k-means iterations for initialization
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
        num_cycles: int = 1,
        ba_iters: int = 2,
        ema_pi_decay: float = 0.99,
        ema_code_decay: float = 0.99,
        chunk_size: int = 8192,
        accept_image_fmap: bool = False,
        channel_last: bool = True,
        use_soft_only: bool = False,
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
        self.num_cycles = num_cycles
        self.ba_iters = ba_iters
        self.ema_pi_decay = ema_pi_decay
        self.ema_code_decay = ema_code_decay
        self.chunk_size = chunk_size
        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last
        self.use_soft_only = use_soft_only
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        # Initialize codebook
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.xavier_uniform_(self.codebook.weight)

        # Initialize beta (inverse temperature, will be annealed during training)
        self.register_buffer('beta', torch.tensor(beta_start))

        # Initialize prior p(k) - uniform initially, updated via EMA as BA marginal
        init_pi = torch.ones(codebook_size) / codebook_size
        self.register_buffer('pi', init_pi)
        
        # Track accumulated soft assignments for BA-style codebook updates
        # These buffers accumulate: numerator = sum_i p(k|e_i) * e_i, denominator = sum_i p(k|e_i)
        self.register_buffer('code_sum', torch.zeros(codebook_size, dim))
        self.register_buffer('code_count', torch.zeros(codebook_size))

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
        """
        Update beta using cosine annealing schedule with optional cycling.

        Beta (inverse temperature) controls the softness of assignments:
        - Low beta (early training): Soft assignments, exploration
        - High beta (late training): Hard assignments, exploitation

        With num_cycles > 1, beta oscillates between beta_start and beta_end
        multiple times, allowing periodic exploration throughout training.

        This follows BA theory: Start with high entropy (soft) to explore the RD space,
        gradually sharpen to converge to optimal quantizer. Cycling allows
        re-exploration of the rate-distortion space.
        """
        if step is None or total_steps is None or total_steps <= 1:
            return

        # Cosine annealing with cycling: smooth transition from beta_start to beta_end
        # repeated num_cycles times
        t = min(step / (total_steps - 1), 1.0)
        cycle_progress = (t * self.num_cycles) % 1.0  # Progress within current cycle
        new_beta = self.beta_start + (self.beta_end - self.beta_start) * 0.5 * (1 - cos(pi * cycle_progress))
        self.beta.data = torch.tensor(new_beta, device=self.beta.device)

    def _compute_distances_chunked(self, z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between inputs and codebook vectors in chunks.
        
        Distance metrics:
        - Euclidean: d(z, c_k) = ||z - c_k||^2
        - Cosine: d(z, c_k) = 1 - cos(z, c_k)
        
        Args:
            z: Input tensor [N, D]
            codebook: Codebook tensor [K, D]

        Returns:
            distances: [N, K] distance matrix
        """
        N = z.shape[0]
        K = codebook.shape[0]
        device = z.device

        # Prepare for distance computation
        if self.use_cosine_sim:
            # Normalize both z and codebook for cosine similarity
            z = l2norm(z)
            codebook = l2norm(codebook)

        # Compute distances in chunks to avoid OOM
        distances = torch.zeros(N, K, device=device)

        for start_idx in range(0, N, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, N)
            z_chunk = z[start_idx:end_idx]  # [chunk_size, D]

            if self.use_cosine_sim:
                # Cosine distance: d = 1 - cos(z, e) = 1 - z·e (normalized)
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

    def _ba_iterations(
        self, 
        distances: torch.Tensor, 
        Q_init: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run Blahut-Arimoto E-M style iterations.
        
        This is the core BA algorithm:
        - E-step: Compute soft assignments p(k|e) ∝ Q(k) * exp(-beta * d(e, c_k))
        - M-step: Update marginal Q(k) = (1/N) * sum_n p(k|e_n)
        
        Theory: These alternations minimize the BA functional and converge to the
        rate-distortion optimal quantizer. Q represents the prior/marginal over codes,
        and p(k|e) are the posterior assignments given encoder outputs.
        
        Args:
            distances: [N, K] distance matrix d(e_i, c_k)
            Q_init: [K] initial Q distribution (prior over codes)

        Returns:
            P: [N, K] soft assignment probabilities p(k|e)
            Q: [K] updated marginal distribution Q(k)
        """
        N, K = distances.shape

        Q = Q_init.clone()

        for ba_iter in range(self.ba_iters):
            # E-step: Compute soft assignments p(k|e)
            # BA formula: p(k|e) ∝ Q(k) * exp(-beta * d(e, c_k))
            # In log space: log p(k|e) = log Q(k) - beta * d(e, c_k) - log Z
            
            log_Q = log(Q)  # [K]
            logits = log_Q.unsqueeze(0) - self.beta * distances  # [N, K]

            # Numerical stability: log-sum-exp trick (subtract max)
            logits_max = logits.max(dim=-1, keepdim=True)[0]
            logits_stable = logits - logits_max

            # Softmax to get probabilities
            P = F.softmax(logits_stable, dim=-1)  # [N, K]

            # M-step: Update marginal Q as average of posteriors
            # BA formula: Q(k) = (1/N) * sum_n p(k|e_n)
            # This is the empirical marginal distribution over codes
            Q = P.mean(dim=0)  # [K]

        return P, Q

    def _update_prior(self, Q: torch.Tensor):
        """
        Update EMA prior p(k) using exponential moving average.
        
        This tracks the dataset-level marginal distribution over codes,
        following BA's optimal rate allocation: codes used frequently
        get higher prior probability.
        """
        if self.training:
            with torch.no_grad():
                self.pi.data = self.ema_pi_decay * self.pi.data + (1 - self.ema_pi_decay) * Q

    def _update_codebook_ba_style(self, e: torch.Tensor, P: torch.Tensor):
        """
        Update codebook using BA-inspired weighted EMA.
        
        BA reproduction update: c_k = sum_i p(k|e_i) * e_i / sum_i p(k|e_i)
        This makes each code the weighted centroid of inputs that softly assign to it.
        
        We use EMA for stability:
        c_k <- (1-alpha) * c_k + alpha * (weighted_centroid)
        
        Theory: This is the M-step for codes in BA, minimizing expected distortion
        under the soft assignment distribution. Codes move toward the center of mass
        of their "soft cluster".
        
        Args:
            e: Encoder outputs [N, D]
            P: Soft assignments [N, K]
        """
        if not self.training:
            return

        with torch.no_grad():
            # Compute weighted centroids: sum_i p(k|e_i) * e_i
            # Shape: [K, D] = einsum([N, K], [N, D])
            code_sum = einsum('n k, n d -> k d', P, e)
            
            # Compute normalization: sum_i p(k|e_i)
            # Shape: [K]
            code_count = P.sum(dim=0)  # [K]
            
            # Normalize to get centroids
            # Add small epsilon to avoid division by zero for unused codes
            weighted_centroids = safe_div(code_sum, code_count.unsqueeze(-1))
            
            # EMA update: c_k <- (1-alpha) * c_k + alpha * centroid_k
            # This smooths updates and provides stability
            alpha = 1.0 - self.ema_code_decay
            self.codebook.weight.data = (
                self.ema_code_decay * self.codebook.weight.data + 
                alpha * weighted_centroids
            )
            
            # For cosine similarity, renormalize after update
            if self.use_cosine_sim:
                self._normalize_codebook()

    def _init_with_kmeans(self, x: torch.Tensor):
        """
        Initialize codebook using k-means clustering on first batch.
        
        This provides a good starting point for the BA algorithm,
        ensuring codes start near the data manifold.
        """
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

    @autocast('cuda', enabled=False)
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Union[Return, Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, dict]]]:
        """
        Forward pass of BA-inspired vector quantization.
        
        Pipeline:
        1. Compute distances d(e, c_k) between encoder outputs and codes
        2. Run BA iterations to get soft assignments p(k|e) and marginal Q(k)
        3. Compute soft quantization: q_soft = sum_k p(k|e) * c_k (fully differentiable!)
        4. Compute hard quantization: q_hard = c_argmax(p(k|e)) (for inference)
        5. Use STE or soft-only mode for gradients
        6. Update codes using BA-style weighted EMA
        7. Compute losses: reconstruction + commitment + codebook + entropy
        
        Args:
            x: Input tensor
                - Image fmaps: [B, C, H, W] if accept_image_fmap=True
                - Sequence: [B, N, D]
                - Flat: [B, D]
            return_all: Return auxiliary metrics dict instead of scalar loss
            step: Current training step (for beta annealing)
            total_steps: Total training steps (for beta annealing)

        Returns:
            If return_all=False:
                (quantized, indices, loss)
            If return_all=True:
                (quantized, indices, aux_dict)
        """
        # Force float32 for numerical stability in distance computations
        x = x.float()

        # Initialize codebook with k-means on first batch
        if self.training:
            self._init_with_kmeans(x)

        # Update beta based on training schedule (cosine annealing)
        if self.training:
            self._update_beta(step, total_steps)

        # Normalize codebook if using cosine similarity
        if self.use_cosine_sim and self.training:
            self._normalize_codebook()

        # Store original shape for later reconstruction
        orig_shape = x.shape
        device = x.device

        # Flatten input to [N, D] format for processing
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

        e = x  # Encoder outputs
        N, D = e.shape

        # Get codebook vectors
        codebook = self.codebook.weight  # [K, D]

        # Compute distance matrix [N, K]
        distances = self._compute_distances_chunked(e, codebook)

        # Run BA iterations to get soft assignments and marginal
        # P: [N, K] soft assignments p(k|e)
        # Q: [K] marginal distribution Q(k)
        P, Q = self._ba_iterations(distances, self.pi)

        # Update prior p(k) using EMA
        self._update_prior(Q)

        # Update codebook using BA-style weighted EMA
        self._update_codebook_ba_style(e, P)

        # === Soft Quantization (Fully Differentiable) ===
        # q_soft = sum_k p(k|e) * c_k
        # This is the BA-style soft quantization: weighted average of codes
        z_q_soft = einsum('n k, k d -> n d', P, codebook)  # [N, D]

        # === Hard Quantization (For Inference) ===
        # Take argmax of soft assignments for discrete indices
        indices = P.argmax(dim=-1)  # [N]
        z_q_hard = F.embedding(indices, codebook)  # [N, D]

        # === Gradient Flow ===
        if self.use_soft_only:
            # Use only soft quantization - fully differentiable, no STE
            # Gradients flow naturally through soft assignments
            z_q = z_q_soft
        else:
            # Straight-through estimator: forward uses hard, backward uses soft
            # This gives us discrete codes at inference while maintaining gradients
            z_q = z_q_soft + (z_q_hard - z_q_soft).detach()

        # === Loss Computation ===
        loss = self.zero.clone()

        # Commitment loss: ||e - sg(z_q_soft)||^2
        # Pulls encoder outputs toward (detached) soft quantized values
        # This encourages encoder to output values close to codebook entries
        commitment_loss = F.mse_loss(e, z_q_soft.detach())
        loss = loss + self.commitment_weight * commitment_loss

        # Codebook loss: ||z_q_soft - sg(e)||^2  
        # Pulls soft quantized values toward (detached) encoder outputs
        # This encourages codebook to move toward the data manifold
        # NOTE: With BA-style EMA updates, this loss might seem redundant,
        # but it provides an additional gradient signal during backprop
        codebook_loss = F.mse_loss(z_q_soft, e.detach())
        loss = loss + self.codebook_weight * codebook_loss

        # Entropy loss: Encourage uniform usage of codebook
        # We use -H(Q) to maximize entropy, encouraging diverse code usage
        # BA theory: High entropy = better rate-distortion trade-off
        H_Q = entropy(Q)
        entropy_loss = -H_Q
        loss = loss + self.entropy_weight * entropy_loss

        # === Compute Metrics ===
        # Perplexity: exp(H(Q)), measures effective codebook size
        # Higher is better (more codes being used)
        perplexity = torch.exp(H_Q)
        
        # Usage: Fraction of codes with non-negligible probability
        # Higher is better (fewer dead codes)
        usage = (Q > 1e-6).float().mean()

        # === Reshape Back to Original Format ===
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

        # === Return Values ===
        if return_all:
            # Return detailed auxiliary metrics for logging/analysis
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
            # Return simple tuple for standard usage
            return z_q, indices_out, loss