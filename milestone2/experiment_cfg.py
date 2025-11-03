from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime

print("Milestone 2: Classifier-Free Guidance Experimentation")
print("="*60)

# Load model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipe.to(device)
print(f"Model loaded on: {device}\n")

# Create output directory
os.makedirs("cfg_experiments", exist_ok=True)

# Test prompt
test_prompt = "A serene Japanese garden with a red bridge over a koi pond"

# Different CFG scale values to test
cfg_scales = [3.0, 5.0, 7.5, 10.0, 15.0]

print(f"Test Prompt: '{test_prompt}'")
print(f"Testing CFG scales: {cfg_scales}\n")

results = []

for cfg in cfg_scales:
    print(f"Generating with CFG scale = {cfg}...")
    
    start_time = datetime.now()
    
    # Generate image
    image = pipe(
        test_prompt, 
        num_inference_steps=20,
        guidance_scale=cfg  # This is the CFG parameter we're testing
    ).images[0]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save image
    filename = f"cfg_experiments/cfg_{cfg}_steps_20.png"
    image.save(filename)
    
    results.append({
        'cfg': cfg,
        'time': duration,
        'file': filename
    })
    
    print(f"  ✓ Saved: {filename} ({duration:.1f}s)\n")

# Summary
print("="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"Prompt: '{test_prompt}'")
print(f"Inference steps: 20")
print("\nResults:")
for r in results:
    print(f"  CFG {r['cfg']:4.1f} → {r['time']:5.1f}s → {r['file']}")

print("\n✓ Experiment complete! Compare images to see CFG effects.")
print("  - Lower CFG (3.0-5.0): More creative, diverse")
print("  - Medium CFG (7.5): Balanced (default)")
print("  - Higher CFG (10.0-15.0): More literal, detailed")