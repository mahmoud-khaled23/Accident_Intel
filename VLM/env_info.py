#!/usr/bin/env python3
"""Print Python / library / GPU environment information."""
import importlib.metadata as md
import platform
import sys

import torch

LIBS = [
    "torch", "torchvision", "torchaudio", "transformers", "accelerate",
    "qwen-vl-utils", "decord", "numpy", "pillow", "huggingface-hub",
    "tokenizers", "safetensors", "opencv-python",
]


def main():
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print()
    for lib in LIBS:
        try:
            print(f"{lib}=={md.version(lib)}")
        except md.PackageNotFoundError:
            print(f"{lib}: not installed")

    print()
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version (torch built with):", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
