"""
Local VQ modules for VQ-VAE experiments.

This module contains:
- VectorQuantize: Standard vector quantization (from the vector-quantize-pytorch library)
- BAVectorQuantize: Blahut-Arimoto inspired vector quantization (our implementation)
"""

from vq_modules.vector_quantize_pytorch import VectorQuantize
from vq_modules.ba_vector_quantize import BAVectorQuantize

__all__ = ['VectorQuantize', 'BAVectorQuantize']
