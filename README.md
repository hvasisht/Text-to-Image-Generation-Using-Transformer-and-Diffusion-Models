# Text-to-Image Generation Using CLIP and Stable Diffusion

This repository contains the implementation and evaluation of a complete text-to-image generation pipeline. The system integrates CLIP text encoding with Stable Diffusion v1.5 for semantic image synthesis.

---

##  Repository Structure
```
milestone1/          → Dataset preparation & baseline generation
├── download_dataset.py       → Download COCO 2017 dataset
├── explore_dataset.py        → Visualize dataset samples
├── test_clip.py              → Validate CLIP text encoding
└── generate_images.py        → Generate 5 baseline images

milestone2/          → Parameter optimization experiments
├── experiment_cfg.py         → Test CFG scales (3.0-15.0)
├── experiment_steps.py       → Test inference steps (10-50)
├── experiment_schedulers.py  → Test schedulers (PNDM, DDIM, LMS, Euler)
├── generate_final_set.py     → Generate 10 optimized images
├── training_log.md           → Detailed experiment observations
└── milestone2_summary.md     → 1-page findings summary

milestone3/          → Quantitative evaluation & metrics
├── prepare_reference_images.py    → Sample 500 COCO images
├── resize_reference_images.py     → Resize to 512×512
├── calculate_fid.py               → FID score calculation
├── calculate_inception_score.py   → Inception Score calculation
├── calculate_clip_similarity.py   → CLIP similarity per image
├── parameter_analysis.py          → Create 4-panel chart
├── visualize_comparison.py        → Create comparison charts
└── milestone3_results.md          → Comprehensive results

demo/                → Interactive web application
└── app.py           → Streamlit interface with 6 advanced features

datasets/            → COCO 2017 validation (gitignored, 1.25GB)
```

---

##  Milestones Summary

### Milestone 1 – Dataset Preparation & Baseline

**Objective:** Establish functional pipeline with CLIP + Stable Diffusion

**Tasks:**
- Downloaded COCO 2017 validation set (5,000 images, 25,014 captions)
- Validated CLIP text encoding (83.24% matching accuracy)
- Generated 5 baseline images with default parameters

**Key Finding:** System functional but shows style inconsistencies

---

### Milestone 2 – Parameter Optimization

**Objective:** Identify optimal generation parameters through systematic experiments

**Experiments Conducted:**

**1. CFG Scale (5 values tested)**
- Optimal: **CFG 7.5** (31s)
- CFG 15.0: 222s (7× slower, minimal gain)

**2. Inference Steps (4 values tested)**
- Optimal: **20 steps** (30s)
- 50 steps: 139s (4.6× slower, marginal improvement)

**3. Noise Schedulers (4 tested)**
- PNDM: 32s (default, reliable)
- DDIM: 30s (fastest, deterministic)

**Final Generation:** 10 diverse images using optimal settings (CFG=7.5, 20 steps)

**Key Finding:** Small parameter changes = massive performance differences (7× speed variation)

---

### Milestone 3 – Quantitative Evaluation

**Objective:** Measure performance with industry-standard metrics

**Metrics Calculated:**

**FID (Fréchet Inception Distance):** 374.47
- Generated: 10 images
- Reference: 500 COCO images (resized 512×512)
- Interpretation: Photorealism limited (baseline model, no fine-tuning)

**Inception Score:** 5.08
- Threshold: >5.0 = Excellent
- Interpretation: Clear, diverse outputs 

**CLIP Similarity:** 31.85 (range: 28.66-34.16)
- Threshold: >0.30 = Excellent
- Interpretation: Perfect semantic alignment 

**Visualizations:** 4 comprehensive charts created

**Key Finding:** Strong understanding + diversity, weak photorealism → baseline model limitation

---

## 🖥️ How to Run

### 1. Setup Environment
```bash
cd Text-to-Image-Generation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset (First Time Only)
```bash
cd milestone1
python download_dataset.py
# Takes 10-20 minutes, downloads 1.25GB
```

### 3. Run Baseline Generation
```bash
# Test CLIP
python test_clip.py

# Generate 5 baseline images
python generate_images.py
# Output: milestone1/generated_images/ (5 images, ~3 min)
```

### 4. Run Optimization Experiments
```bash
cd ../milestone2

# CFG experiment (5 images)
python experiment_cfg.py

# Steps experiment (4 images)
python experiment_steps.py

# Scheduler experiment (4 images)
python experiment_schedulers.py

# Generate final optimized set (10 images)
python generate_final_set.py
```

### 5. Calculate Metrics
```bash
cd ../milestone3

# Prepare reference data
python prepare_reference_images.py
python resize_reference_images.py

# Calculate all metrics
python calculate_fid.py          # FID: 374.47
python calculate_inception_score.py  # IS: 5.08
python calculate_clip_similarity.py  # CLIP: 31.85

# Create visualizations
python parameter_analysis.py
python visualize_comparison.py
```

### 6. Run Interactive Demo
```bash
cd ../demo
streamlit run app.py
# Opens browser at localhost:8501
```

---

##  Demo Features (Beyond Requirements)

1. **Spell Correction** - Auto-fixes typos (TextBlob)
2. **Style Presets** - 7 artistic styles (Photorealistic, Anime, Oil Painting, etc.)
3. **Negative Prompts** - Specify unwanted elements
4. **Quality Scoring** - Real-time CLIP similarity feedback
5. **Batch Generation** - Create 1-4 variations
6. **Prompt History** - Reuse previous prompts

---

##  Results Summary

### Parameter Optimization

| Parameter | Tested Values | Optimal | Speedup |
|-----------|--------------|---------|---------|
| CFG Scale | 3.0-15.0 | 7.5 | 7× vs CFG 15 |
| Inference Steps | 10-50 | 20 | 4.6× vs 50 steps |
| Scheduler | 4 types | PNDM/DDIM | Minimal impact |

### Performance Metrics

- **Generation Time:** 35-40s avg (Apple Silicon MPS)
- **Resolution:** 512×512
- **Hardware:** Mac (MPS acceleration)
- **Sample Size:** 10 final images, 23 experimental images

### Evaluation Scores

- **FID:** 374.47 (needs photorealism improvement)
- **IS:** 5.08 (excellent clarity + diversity)
- **CLIP:** 31.85 (perfect semantic alignment)

---

##  Known Limitations

- **Style inconsistency** - Some images render in cartoon/illustration style
- **Small sample** - 10 images insufficient for robust FID
- **Resolution** - Limited to 512×512 on consumer hardware
- **Speed** - 35s on Mac vs 5-10s on A100 GPU

**Root Cause:** Baseline Stable Diffusion trained on mixed data (photos + art + cartoons) without fine-tuning

---

## Future Improvements

- Fine-tune on COCO for photorealism (expect FID: 100-150)
- Generate 100+ images for robust metrics
- Higher resolution (768×768, 1024×1024)
- ControlNet for spatial control
- Faster generation (<10s)

---

##  Team 

- **Harini Prasad Vasisht** - vasisht.h@northeastern.edu
- **Samruddhi Bansod** - bansod.s@northeastern.edu
- **Pranav Rangbulla** - rangbulla.p@northeastern.edu
- **Dhanush Manoharan** - manoharan.d@northeastern.edu

---

## 📄 Citation
```bibtex
@project{vasisht2025texttoimage,
  title={Text-to-Image Generation Using CLIP and Stable Diffusion},
  author={Vasisht, Harini Prasad and Bansod, Samruddhi and Rangbulla, Pranav and Manoharan, Dhanush},
  institution={Northeastern University},
  course={IE 7615 Deep Learning for AI},
  year={2025}
}
```

---

##  References

[1] Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)  
[2] Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (2021)  
[3] Schuhmann et al. "LAION-5B: An Open Large-Scale Dataset" (2022)  
[4] Lin et al. "Microsoft COCO: Common Objects in Context" (2014)

---

