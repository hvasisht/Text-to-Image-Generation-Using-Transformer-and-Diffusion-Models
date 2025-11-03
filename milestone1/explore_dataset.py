import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random

# Load captions
captions_path = "../datasets/coco_2017/annotations/captions_val2017.json"
with open(captions_path, 'r') as f:
    coco_data = json.load(f)

print(f"Total images: {len(coco_data['images'])}")
print(f"Total captions: {len(coco_data['annotations'])}")
print(f"\nSample image info:")
print(coco_data['images'][0])
print(f"\nSample caption:")
print(coco_data['annotations'][0])

# Function to display image with its captions
def show_image_with_captions(image_id, num_captions=5):
    # Find image info
    img_info = next(img for img in coco_data['images'] if img['id'] == image_id)
    
    # Find all captions for this image
    captions = [ann['caption'] for ann in coco_data['annotations'] 
                if ann['image_id'] == image_id][:num_captions]
    
    # Load and display image
    img_path = f"../datasets/coco_2017/val2017/{img_info['file_name']}"
    img = Image.open(img_path)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Sample Captions:\n" + "\n".join([f"{i+1}. {cap}" for i, cap in enumerate(captions)]), 
              fontsize=10, loc='left')
    plt.tight_layout()
    plt.show()

# Show 3 random samples
print("\n\nDisplaying 3 random images with captions...")
random_image_ids = random.sample([img['id'] for img in coco_data['images']], 3)
for img_id in random_image_ids:
    show_image_with_captions(img_id)