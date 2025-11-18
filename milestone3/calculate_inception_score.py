import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
from scipy.stats import entropy

if __name__ == '__main__':
    print("Calculating Inception Score")
    print("="*60)

    # Load Inception V3 model
    print("Loading Inception V3 model...")
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    inception_model = inception_model.to(device)

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load generated images
    gen_dir = Path("../milestone2/final_images")
    images = list(gen_dir.glob("*.png"))
    
    print(f"Found {len(images)} generated images")
    print("\nProcessing images...\n")

    predictions = []
    
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = inception_model(img_tensor)
            pred = nn.functional.softmax(pred, dim=1)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Calculate Inception Score
    # Split predictions into groups
    split_scores = []
    splits = 1  # Using 1 split for small dataset
    
    for k in range(splits):
        part = predictions[k * (len(predictions) // splits): (k + 1) * (len(predictions) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    inception_score = np.mean(split_scores)
    std = np.std(split_scores)
    
    print("="*60)
    print("INCEPTION SCORE RESULTS")
    print("="*60)
    print(f"Inception Score: {inception_score:.2f} ± {std:.2f}")
    print("\nInterpretation:")
    print("  > 5.0 = Excellent")
    print("  3.0-5.0 = Good")
    print("  2.0-3.0 = Acceptable")
    print("  < 2.0 = Needs improvement")
    print("\n✓ Inception Score calculation complete!")
    
    # Save results
    with open("inception_score_results.txt", "w") as f:
        f.write(f"Inception Score: {inception_score:.2f} ± {std:.2f}\n")
        f.write(f"Number of images: {len(images)}\n")
    
    print(f"\nResults saved to: inception_score_results.txt")