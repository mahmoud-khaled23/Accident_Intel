"""Download the parts-segmentation weights into ./weights and print its classes.

This script fetches the damage-segmentation model from Hugging Face Hub,
saves it to the local ``weights/`` directory, and lists the class names the
model was trained to detect.  Run it once before using the pipeline if the
weights file is not already present.

Usage::

    python download_weights.py
"""
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

Path("weights").mkdir(exist_ok=True)

# path = hf_hub_download(
#     repo_id="mitbersh/car-parts-segmentation",
#     filename="parts_segmentation.pt",
#     local_dir="weights",
# )
# print("saved to:", path)

# model = YOLO(path)
# print("classes:")
# for i, name in model.names.items():
#     print(f"  {i}: {name}")

# model_path = hf_hub_download(
#     repo_id="abdullahg7/cardd-yolov8s",
#     filename="v2.0/best.pt",
#     local_dir="weights",
# )

# model = YOLO(model_path)
# for i, name in model.names.items():
#     print(f"  {i}: {name}")

path = hf_hub_download(
    repo_id = "mitbersh/car-damage-segmentation",
    filename="damage_segmentation.pt",
    local_dir="weights",
)

model = YOLO(path)
print("classes:")
for i, name in model.names.items():
    print(f"  {i}: {name}")
