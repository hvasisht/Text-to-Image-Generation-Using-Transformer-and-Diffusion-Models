# Milestone 2: Summary of Observations

**Course:** IE 7615 Deep Learning for AI  
**Student:** Harini Vasisht  
**Date:** November 3, 2025

---

## Objective

Milestone 2 focused on optimizing the text-to-image generation pipeline by systematically tuning three key parameters: classifier-free guidance (CFG) scale, inference steps, and noise schedulers. The goal was to identify optimal settings that balance image quality, semantic alignment, and generation speed.

---

## Experiments Conducted

### 1. Classifier-Free Guidance (CFG) Scale Testing
**Range tested:** 3.0, 5.0, 7.5, 10.0, 15.0  
**Key finding:** CFG 7.5 provides optimal balance. Higher values (15.0) dramatically increase generation time (7x slower at 222.5s vs 30-31s) with minimal quality improvement. Lower values (3.0-5.0) produce more creative but less prompt-accurate results.

### 2. Inference Steps Optimization  
**Range tested:** 10, 20, 30, 50 steps  
**Key finding:** 20 steps offer best quality-to-speed ratio (30.0s). 50 steps provide marginal quality improvement but take 4.6x longer (139.4s). Diminishing returns observed after 30 steps.

### 3. Noise Scheduler Comparison
**Schedulers tested:** PNDM, DDIM, LMS, Euler  
**Key finding:** DDIM scheduler slightly faster (29.9s) than default PNDM (31.9s) with comparable quality. Scheduler choice has less impact than CFG and inference steps on final output quality.

---

## Optimal Settings Identified

Based on experimental results:
- **CFG Scale:** 7.5 (balanced prompt adherence and generation speed)
- **Inference Steps:** 20 (optimal quality-to-speed ratio)
- **Scheduler:** PNDM or DDIM (both perform well, DDIM marginally faster)

These settings were used to generate the final set of 10 diverse images covering various scene types.

---

## Results Summary

**Total images generated:** 10  
**Average generation time:** 41.2 seconds per image  
**Hardware:** Apple Silicon (MPS acceleration)

All 10 images demonstrated strong semantic alignment with their text prompts across diverse categories:
- Nature scenes (mountains, forests, beaches, lavender fields)
- Animals (lion in savanna)
- Architecture (castle, library)
- Urban environments (Tokyo night market)
- Science fiction (astronaut in space)
- Fantasy (steampunk robot)

---

## Observations and Limitations

### Strengths:
- Consistent generation across diverse prompt types
- Good semantic understanding and object placement
- Realistic textures and lighting in most scenes
- Stable performance without memory issues

### Limitations Observed:
1. **Style inconsistency:** Some images showed mixed rendering styles (e.g., Image 01 - lion's face appeared more stylized/animated while body remained photorealistic)
2. **Fine details:** Small objects and intricate patterns occasionally lacked sharp detail at 512×512 resolution
3. **Generation time variability:** Complex prompts with multiple elements took longer (up to 50s vs 30s for simpler scenes)
4. **Text rendering:** Any text within generated images (e.g., signs) appeared as non-readable characters (known limitation of diffusion models)

---

## Key Insights

1. **Parameter sensitivity:** CFG scale has the most significant impact on both quality and speed. Values above 10.0 provide diminishing returns with dramatic speed penalties.

2. **Hardware performance:** MPS acceleration on Apple Silicon proved adequate for development work, though generation times are slower than high-end GPU setups.

3. **Trade-offs:** Clear trade-off exists between generation speed and quality. For iterative development, 20 steps is optimal. For final production outputs, 30 steps may be preferable if time permits.

4. **Baseline model capabilities:** Pre-trained Stable Diffusion v1.5 demonstrates strong zero-shot generalization without fine-tuning, validating the approach for this project.

---

## Challenges Encountered

1. **Dependency management:** Required installation of scipy library for LMS scheduler compatibility
2. **Time management:** High CFG values (15.0) and step counts (50) significantly extended experiment duration
3. **Qualitative assessment:** Evaluating "quality" remains subjective without quantitative metrics (to be addressed in Milestone 3)

---

## Next Steps (Milestone 3)

1. **Quantitative evaluation:** Implement FID (Fréchet Inception Distance) and Inception Score metrics to objectively measure image quality
2. **Systematic comparison:** Compare outputs across different parameter combinations using quantitative metrics
3. **Semantic alignment analysis:** Evaluate text-image correspondence using CLIP similarity scores
4. **Parameter sensitivity study:** Document how small parameter changes affect output metrics

---

## Conclusion

Milestone 2 successfully identified optimal parameter settings (CFG=7.5, Steps=20) for the text-to-image generation pipeline. The experiments demonstrated clear trade-offs between quality, speed, and prompt adherence. Generated images showed strong semantic alignment with diverse prompts, though some limitations (style inconsistency, detail resolution) were observed. These findings establish a solid baseline for quantitative evaluation in Milestone 3.

**Status:** ✅ Milestone 2 Complete  
**Deliverables:** Training log, 10 sample images, observations summary  
**Next Milestone:** Evaluation & Analysis (Week 5)