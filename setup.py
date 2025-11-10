"""
Setup script for vq_modules - Local VQ implementations for VQ-VAE experiments
"""

from setuptools import setup, find_packages

setup(
    name="vq_modules",
    version="0.1.0",
    description="Vector Quantization modules including Blahut-Arimoto VQ",
    author="Nikita Karagodin",
    packages=find_packages(include=['vq_modules', 'vq_modules.*']),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'einops>=0.6.0',
    ],
)
