from setuptools import setup, find_packages

setup(
    name="adapt-hrl",
    version="1.0.0",
    description=(
        "AdaptHRL: Adaptive Recursive Hierarchical Decomposition "
        "for Scalable Long-Horizon Decision-Making (NeurIPS 2025)"
    ),
    author="[S M Jubaer]",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "gymnasium>=0.29.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "mujoco": ["mujoco==2.3.7", "d4rl>=1.1"],
        "metaworld": ["metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git"],
        "dev": ["pytest>=7.0", "black", "isort", "flake8"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
