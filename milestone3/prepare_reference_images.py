import os
import shutil
from pathlib import Path
import random

print("Preparing Reference Images for FID Calculation")
print("="*60)

# Paths
coco_images = Path("../datasets/coco_2017/val2017")
output_dir = Path("reference_images")

# Create output directory
output_dir.mkdir(exist_ok=True)

# Get all COCO images
all_images = list(coco_images.glob("*.jpg"))
print(f"Total COCO images available: {len(all_images)}")

# Select random subset (we'll use 500 images for comparison)
num_reference = 500
selected_images = random.sample(all_images, num_reference)

print(f"\nCopying {num_reference} random images as reference set...")

# Copy selected images
for i, img_path in enumerate(selected_images, 1):
    dest = output_dir / img_path.name
    shutil.copy(img_path, dest)
    if i % 100 == 0:
        print(f"  Copied {i}/{num_reference} images...")

print(f"\n✓ Reference set ready!")
print(f"Location: {output_dir.absolute()}")
print(f"Total images: {len(list(output_dir.glob('*.jpg')))}")