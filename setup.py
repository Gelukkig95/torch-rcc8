from setuptools import find_packages, setup
setup(
    name="torch-rcc8",
    version="0.2.0",
    description="Differentiable Raster & Continuous-SDF RCC-8 in PyTorch",
    author="Jinwon Kim",
    packages=find_packages(include=["torch_rcc8", "torch_rcc8.*"]),
    python_requires=">=3.9",
    install_requires=["torch>=2.0.0", "numpy>=1.23.0"],
)
