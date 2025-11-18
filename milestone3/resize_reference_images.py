from PIL import Image
from pathlib import Path
from tqdm import tqdm

print("Resizing Reference Images to 512x512")
print("="*60)

ref_dir = Path("reference_images")
output_dir = Path("reference_images_resized")
output_dir.mkdir(exist_ok=True)

# Get all JPG images
images = list(ref_dir.glob("*.jpg"))
print(f"Found {len(images)} images to resize\n")

for img_path in tqdm(images, desc="Resizing"):
    # Open and resize
    img = Image.open(img_path)
    img_resized = img.resize((512, 512), Image.LANCZOS)
    
    # Save
    output_path = output_dir / img_path.name
    img_resized.save(output_path)

print(f"\n✓ All images resized to 512x512")
print(f"Location: {output_dir}")