from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler
import torch
import os
from datetime import datetime

print("Milestone 2: Noise Scheduler Experimentation")
print("="*60)

# Load base model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

device = "mps" if torch.backends.mps.is_available() else "cpu"
pipe = pipe.to(device)
print(f"Model loaded on: {device}\n")

# Create output directory
os.makedirs("scheduler_experiments", exist_ok=True)

# Test prompt
test_prompt = "A futuristic city skyline at sunset with flying cars"

# Different schedulers to test
schedulers = {
    'PNDM': PNDMScheduler.from_config(pipe.scheduler.config),
    'DDIM': DDIMScheduler.from_config(pipe.scheduler.config),
    'LMS': LMSDiscreteScheduler.from_config(pipe.scheduler.config),
    'Euler': EulerDiscreteScheduler.from_config(pipe.scheduler.config)
}

print(f"Test Prompt: '{test_prompt}'")
print(f"Testing schedulers: {list(schedulers.keys())}")
print(f"Inference steps: 20, CFG scale: 7.5\n")

results = []

for name, scheduler in schedulers.items():
    print(f"Generating with {name} scheduler...")
    
    # Set the scheduler
    pipe.scheduler = scheduler
    
    start_time = datetime.now()
    
    # Generate image
    image = pipe(
        test_prompt, 
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save image
    filename = f"scheduler_experiments/scheduler_{name}.png"
    image.save(filename)
    
    results.append({
        'scheduler': name,
        'time': duration,
        'file': filename
    })
    
    print(f"  ✓ Saved: {filename} ({duration:.1f}s)\n")

# Summary
print("="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"Prompt: '{test_prompt}'")
print(f"Inference steps: 20, CFG: 7.5")
print("\nResults:")
for r in results:
    print(f"  {r['scheduler']:6s} → {r['time']:5.1f}s → {r['file']}")

print("\n✓ Experiment complete! Compare images to see scheduler differences.")
print("  - PNDM: Default scheduler (what we've been using)")
print("  - DDIM: Deterministic, good quality")
print("  - LMS: Smoother, can be slower")
print("  - Euler: Fast, good for lower step counts")