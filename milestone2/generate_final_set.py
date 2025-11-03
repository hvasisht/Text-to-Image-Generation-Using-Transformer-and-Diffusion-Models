from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime

print("Milestone 2: Final Image Set Generation")
print("="*60)

# Load model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipe.to(device)
print(f"Model loaded on: {device}\n")

# Create output directory
os.makedirs("final_images", exist_ok=True)

# Optimal settings based on experiments
optimal_cfg = 7.5
optimal_steps = 20

print(f"Using optimal settings: CFG={optimal_cfg}, Steps={optimal_steps}\n")

# 10 diverse prompts covering different categories
prompts = [
    "A majestic lion resting under an acacia tree in the African savanna",
    "An astronaut floating in space with Earth in the background",
    "A medieval castle on a cliff overlooking the ocean at dawn",
    "A bustling night market in Tokyo with colorful neon signs",
    "A peaceful mountain lake reflecting snow-capped peaks",
    "An old library filled with ancient books and warm lamplight",
    "A field of lavender flowers stretching to the horizon under blue sky",
    "A steampunk robot playing chess in a Victorian study",
    "A tropical beach with turquoise water and palm trees at golden hour",
    "An autumn forest path covered with red and orange leaves"
]

results = []

for i, prompt in enumerate(prompts, 1):
    print(f"[{i}/10] Generating: '{prompt[:50]}...'")
    
    start_time = datetime.now()
    
    # Generate image
    image = pipe(
        prompt, 
        num_inference_steps=optimal_steps,
        guidance_scale=optimal_cfg
    ).images[0]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save image
    filename = f"final_images/image_{i:02d}.png"
    image.save(filename)
    
    results.append({
        'id': i,
        'prompt': prompt,
        'time': duration,
        'file': filename
    })
    
    print(f"    ✓ Saved: {filename} ({duration:.1f}s)\n")

# Summary
print("="*60)
print("FINAL IMAGE SET COMPLETE")
print("="*60)
print(f"Total images: {len(results)}")
print(f"Settings: CFG={optimal_cfg}, Steps={optimal_steps}")
print(f"Total time: {sum(r['time'] for r in results):.1f}s")
print(f"Average time: {sum(r['time'] for r in results)/len(results):.1f}s per image")
print("\nGenerated images:")
for r in results:
    print(f"  {r['id']:2d}. {r['file']}")

print("\n✓ All images generated successfully!")
print(f"Check the 'final_images' folder to view all outputs.")