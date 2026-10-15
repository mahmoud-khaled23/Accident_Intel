#!/usr/bin/env python3
"""Print Python / library / GPU environment information.

Useful for diagnosing dependency or hardware issues before running the
Accident Intel pipeline.  Checks every library listed in :data:`LIBS` and
reports whether CUDA is available.

Key function: :func:`main`.
"""
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
    """Print installed versions of key libraries and GPU availability.

    Iterates over :data:`LIBS`, querying ``importlib.metadata`` for each
    installed package version.  Packages that are not installed are reported
    as "not installed" rather than raising an exception.  After the library
    table, prints CUDA availability and the GPU device name when a GPU is
    present.

    Returns:
        None
    """
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
