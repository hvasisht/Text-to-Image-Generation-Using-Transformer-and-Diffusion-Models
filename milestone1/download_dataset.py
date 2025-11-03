import os
import json
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

# Create directories
data_dir = Path("../datasets/coco_2017")
images_dir = data_dir / "val2017"
anno_dir = data_dir / "annotations"
images_dir.mkdir(parents=True, exist_ok=True)
anno_dir.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("COCO 2017 Dataset Download")
print("=" * 50)

# Download function with progress bar
def download_file(url, dest_path):
    print(f"\nDownloading: {url}")
    print(f"Destination: {dest_path}")
    
    urllib.request.urlretrieve(url, dest_path)
    print("✓ Download complete!")

# 1. Download validation images (only 1GB - smaller than train set)
val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
val_images_zip = data_dir / "val2017.zip"

print("\n[1/2] Downloading validation images (~1GB)...")
print("This will take 5-15 minutes depending on your internet speed.")
download_file(val_images_url, val_images_zip)

# 2. Download annotations
anno_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
anno_zip = data_dir / "annotations_trainval2017.zip"

print("\n[2/2] Downloading annotations (~252MB)...")
download_file(anno_url, anno_zip)

# Extract files
print("\n" + "=" * 50)
print("Extracting files...")
print("=" * 50)

print("\nExtracting validation images...")
with zipfile.ZipFile(val_images_zip, 'r') as zip_ref:
    zip_ref.extractall(data_dir)
print("✓ Images extracted!")

print("\nExtracting annotations...")
with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
    zip_ref.extractall(data_dir)
print("✓ Annotations extracted!")

# Clean up zip files
print("\nCleaning up zip files...")
val_images_zip.unlink()
anno_zip.unlink()

# Verify download
captions_file = anno_dir / "captions_val2017.json"
with open(captions_file, 'r') as f:
    captions_data = json.load(f)

print("\n" + "=" * 50)
print("DOWNLOAD COMPLETE!")
print("=" * 50)
print(f"Total images: {len(captions_data['images'])}")
print(f"Total captions: {len(captions_data['annotations'])}")
print(f"Location: {data_dir.absolute()}")