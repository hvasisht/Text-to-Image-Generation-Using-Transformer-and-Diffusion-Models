# Milestone 3: Evaluation Results Summary

**Student:** Harini Vasisht | **Date:** November 3, 2025

---

## Quantitative Metrics Calculated

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **FID Score** | 374.47 | Needs improvement (photorealism issues) |
| **Inception Score** | 5.08 | Excellent (clear, diverse images) |
| **CLIP Similarity** | 31.85 | Excellent (strong text-image alignment) |

---

## Key Findings

**Good News:**
- ✅ Images match their text prompts very well (CLIP: 31.85)
- ✅ Images are clear and diverse (IS: 5.08)
- ✅ All 10 images showed consistent quality

**Needs Improvement:**
- ❌ Not photorealistic enough (FID: 374.47)
- ❌ Style inconsistencies (cartoon astronaut, animated lion)
- ❌ Baseline model limitation - no fine-tuning

**Why This Happened:**
Model trained on mixed data (photos + cartoons + art). Good at understanding meaning but mixes styles.

---

## Parameter Analysis Results (from Milestone 2)

**Optimal Settings Confirmed:**
- CFG = 7.5 (best balance)
- Steps = 20 (best speed/quality)
- Scheduler = PNDM/DDIM (both work well)

**Key Observations:**
- CFG 15.0 = 7x slower, minimal quality gain
- 50 steps = 4.6x slower, diminishing returns
- Schedulers matter less than CFG and steps

---

## Visualizations Created

1. `parameter_analysis.png` - 4-panel comparison
2. `time_comparison.png` - Speed comparison chart

---

## Conclusion

System works well for semantic understanding and diversity but needs fine-tuning for photorealism. Optimal parameters identified: CFG=7.5, Steps=20.

**Status:** ✅ Complete