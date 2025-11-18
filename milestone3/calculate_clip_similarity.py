import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path

if __name__ == '__main__':
    print("Calculating CLIP Similarity (Text-Image Alignment)")
    print("="*60)

    # Load CLIP
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on: {device}\n")

    # Original prompts from milestone2
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

    gen_dir = Path("../milestone2/final_images")
    images = sorted(list(gen_dir.glob("*.png")))
    
    print("Calculating similarity scores...\n")
    
    results = []
    
    for i, (img_path, prompt) in enumerate(zip(images, prompts), 1):
        # Load image
        image = Image.open(img_path)
        
        # Process
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get similarity
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarity = logits_per_image[0][0].item()
        
        results.append({
            'image': img_path.name,
            'prompt': prompt,
            'similarity': similarity
        })
        
        print(f"[{i:2d}] {img_path.name}: {similarity:.4f}")
    
    # Calculate average
    avg_similarity = sum(r['similarity'] for r in results) / len(results)
    
    print("\n" + "="*60)
    print("CLIP SIMILARITY RESULTS")
    print("="*60)
    print(f"Average Similarity: {avg_similarity:.4f}")
    print("\nInterpretation:")
    print("  > 0.30 = Excellent alignment")
    print("  0.25-0.30 = Good alignment")
    print("  0.20-0.25 = Acceptable alignment")
    print("  < 0.20 = Poor alignment")
    print("\n✓ CLIP similarity calculation complete!")
    
    # Save results
    with open("clip_similarity_results.txt", "w") as f:
        f.write(f"Average CLIP Similarity: {avg_similarity:.4f}\n\n")
        f.write("Individual Results:\n")
        for r in results:
            f.write(f"{r['image']}: {r['similarity']:.4f}\n")
    
    print(f"\nResults saved to: clip_similarity_results.txt")