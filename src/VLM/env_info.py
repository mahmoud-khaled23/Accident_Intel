#!/usr/bin/env python3
"""Print Python / library / GPU environment information for Lightning AI and local setups."""
import importlib.metadata as md
import platform
import sys

import torch

LIBS = [
    "torch", "torchvision", "torchaudio", "transformers", "accelerate",
    "qwen-vl-utils", "decord", "numpy", "pillow", "huggingface-hub",
    "tokenizers", "safetensors", "opencv-python", "opencv-python-headless",
    "bitsandbytes", "flash-attn",
]


def main():
    print("=" * 60)
    print("Environment Diagnostics")
    print("=" * 60)
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print()
    print("Installed Libraries:")
    for lib in LIBS:
        try:
            print(f"  {lib} == {md.version(lib)}")
        except md.PackageNotFoundError:
            print(f"  {lib}: not installed")

    print()
    print("CUDA & GPU Status:")
    print("  CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  CUDA version (torch built with):", torch.version.cuda)
        print("  Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_vram_gb = props.total_memory / (1024 ** 3)
            print(f"  GPU {i}: {props.name} | VRAM: {total_vram_gb:.2f} GB | Compute: {props.major}.{props.minor}")
        print("  bfloat16 supported:", torch.cuda.is_bf16_supported())
    else:
        print("  WARNING: No CUDA GPU detected. VLM inference will be very slow or run out of RAM.")
    print("=" * 60)


if __name__ == "__main__":
    main()

