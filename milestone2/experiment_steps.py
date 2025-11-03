from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime

print("Milestone 2: Inference Steps Experimentation")
print("="*60)

# Load model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipe.to(device)
print(f"Model loaded on: {device}\n")

# Create output directory
os.makedirs("steps_experiments", exist_ok=True)

# Test prompt
test_prompt = "A cozy coffee shop interior with warm lighting and wooden furniture"

# Different inference steps to test
step_counts = [10, 20, 30, 50]

print(f"Test Prompt: '{test_prompt}'")
print(f"Testing inference steps: {step_counts}")
print(f"CFG scale: 7.5 (default)\n")

results = []

for steps in step_counts:
    print(f"Generating with {steps} inference steps...")
    
    start_time = datetime.now()
    
    # Generate image
    image = pipe(
        test_prompt, 
        num_inference_steps=steps,  # This is what we're testing
        guidance_scale=7.5  # Keep CFG constant
    ).images[0]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save image
    filename = f"steps_experiments/steps_{steps}_cfg_7.5.png"
    image.save(filename)
    
    results.append({
        'steps': steps,
        'time': duration,
        'file': filename
    })
    
    print(f"  ✓ Saved: {filename} ({duration:.1f}s)\n")

# Summary
print("="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"Prompt: '{test_prompt}'")
print(f"CFG scale: 7.5")
print("\nResults:")
for r in results:
    print(f"  Steps {r['steps']:2d} → {r['time']:5.1f}s → {r['file']}")

print("\n✓ Experiment complete! Compare images to see quality vs speed trade-off.")
print("  - 10 steps: Fastest but lower quality")
print("  - 20 steps: Balanced (our baseline)")
print("  - 30 steps: Better quality, slower")
print("  - 50 steps: Highest quality, slowest")