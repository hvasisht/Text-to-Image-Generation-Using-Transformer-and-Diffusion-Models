from pytorch_fid import fid_score
import torch
from pathlib import Path

if __name__ == '__main__':
    print("Calculating FID (Fréchet Inception Distance)")
    print("="*60)

    # Paths
    generated_images = Path("../milestone2/final_images")
    reference_images = Path("reference_images_resized")  # Changed from "reference_images"

    print(f"Generated images: {generated_images}")
    print(f"Reference images: {reference_images}")

    # Count images
    gen_count = len(list(generated_images.glob("*.png")))
    ref_count = len(list(reference_images.glob("*.jpg")))

    print(f"\nGenerated: {gen_count} images")
    print(f"Reference: {ref_count} images")

    if gen_count < 2 or ref_count < 2:
        print("\n❌ Error: Need at least 2 images in each folder")
        exit(1)

    print("\nCalculating FID score...")
    print("(This may take 2-3 minutes...)\n")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Calculate FID with num_workers=0 to avoid multiprocessing issues
    fid_value = fid_score.calculate_fid_given_paths(
        [str(generated_images), str(reference_images)],
        batch_size=8,
        device=device,
        dims=2048,
        num_workers=0  # Fix for Python 3.13 multiprocessing issue
    )

    print("="*60)
    print("FID SCORE RESULTS")
    print("="*60)
    print(f"FID Score: {fid_value:.2f}")
    print("\nInterpretation:")
    print("  < 50  = Excellent (state-of-the-art models)")
    print("  50-100 = Good")
    print("  100-150 = Acceptable")
    print("  > 150 = Needs improvement")
    print("\n✓ FID calculation complete!")

    # Save results
    with open("fid_results.txt", "w") as f:
        f.write(f"FID Score: {fid_value:.2f}\n")
        f.write(f"Generated images: {gen_count}\n")
        f.write(f"Reference images: {ref_count}\n")

    print(f"\nResults saved to: fid_results.txt")