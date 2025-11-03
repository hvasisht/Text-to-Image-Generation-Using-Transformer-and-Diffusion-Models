import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import json

print("Loading CLIP model...")
# Load pre-trained CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("✓ CLIP model loaded successfully!")

# Test with some sample text prompts
test_prompts = [
    "A photo of a cat",
    "A beautiful sunset over the ocean",
    "People sitting in a restaurant",
    "A city street at night"
]

print("\nGenerating text embeddings for sample prompts...")
for prompt in test_prompts:
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    text_embeddings = model.get_text_features(**inputs)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Embedding shape: {text_embeddings.shape}")
    print(f"Embedding (first 5 values): {text_embeddings[0][:5].detach().numpy()}")

# Test with an actual image from our dataset
print("\n" + "="*50)
print("Testing CLIP with image + text matching...")
print("="*50)

# Load a sample image
with open("../datasets/coco_2017/annotations/captions_val2017.json", 'r') as f:
    coco_data = json.load(f)

img_info = coco_data['images'][0]
img_path = f"../datasets/coco_2017/val2017/{img_info['file_name']}"
image = Image.open(img_path)

# Get captions for this image
img_captions = [ann['caption'] for ann in coco_data['annotations'] 
                if ann['image_id'] == img_info['id']][:3]

print(f"\nImage: {img_info['file_name']}")
print(f"Captions: {img_captions}")

# Process image and text
inputs = processor(text=img_captions, images=image, return_tensors="pt", padding=True)

# Get embeddings
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print(f"\nSimilarity scores between image and captions:")
for i, caption in enumerate(img_captions):
    print(f"  Caption {i+1}: {probs[0][i].item():.4f} - '{caption}'")

print("\n✓ CLIP text→embedding pipeline working!")