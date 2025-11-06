from setuptools import setup, find_packages

setup(
    name="lora_minisam",
    version="0.1.0",
    description="LoRA-Adapted Lightweight SAM for Medical Image Segmentation",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "opencv-python",
        "pillow",
        "numpy",
        "tqdm",
        "hydra-core>=1.3",
        "pandas",
        "scikit-image",
        "matplotlib",
    ],
    extras_require={
        "dev": ["black", "flake8", "pytest", "jupyter"],
    },
    entry_points={
        "console_scripts": [
            "lora-minisam-preprocess=data.cosmo1050k.data_preprocessing:main",
            "lora-minisam-eval=src.inference.baseline_eval:main",
        ],
    },
)