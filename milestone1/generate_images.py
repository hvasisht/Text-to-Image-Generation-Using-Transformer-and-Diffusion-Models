import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

print("Loading Stable Diffusion model...")
print("(This will download ~4GB and take 5-10 minutes on first run)")

# Load Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Use CPU or MPS (for Mac)
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipe.to(device)

print(f"✓ Model loaded on device: {device}")

# Create output directory
os.makedirs("generated_images", exist_ok=True)

# Test prompts for Milestone 1
test_prompts = [
    "A photo of a cat sitting on a windowsill",
    "A beautiful sunset over the ocean",
    "People sitting in a restaurant having dinner",
    "A city street at night with neon lights",
    "A mountain landscape with snow"
]

print("\n" + "="*50)
print("Generating 5 sample images...")
print("="*50)

for i, prompt in enumerate(test_prompts):
    print(f"\n[{i+1}/5] Generating: '{prompt}'")
    
    # Generate image
    image = pipe(prompt, num_inference_steps=20).images[0]
    
    # Save image
    output_path = f"generated_images/sample_{i+1}.png"
    image.save(output_path)
    
    print(f"✓ Saved to: {output_path}")

print("\n" + "="*50)
print("✓ All 5 images generated successfully!")
print(f"Check the 'generated_images' folder")
print("="*50)