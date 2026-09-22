# ADR 0004: Dual Scoring Evaluation — Official Composite vs Continuous Audited

## Status
Accepted

## Context
In early benchmark iterations, a composite weighted metric was employed to rank model performance:
$$\text{Composite} = 0.4 \times S_{\text{speed}} + 0.3 \times S_{\text{mem}} + 0.3 \times S_{\text{acc}}$$
where:
$$S_{\text{speed}} = \min\left(\frac{1/\text{RTF}}{15}, 1.0\right), \quad S_{\text{mem}} = \min\left(\frac{8192}{\text{Peak MB}}, 1.0\right), \quad S_{\text{acc}} = \max\left(0, 1 - \frac{\text{normWER}}{0.30}\right)$$

### The Saturation Failure
Audit by DeepInvestigator revealed severe clipping saturation:
1. **Speed Ceiling Saturation**: Any model exceeding $15\times$ real-time received $S_{\text{speed}} = 1.0$. Consequently, `Parakeet-TDT-0.6B` ($72.99\times$ real-time) was treated as having identical speed utility to a $16\times$ or $18\times$ model.
2. **Memory Ceiling Saturation**: The 8,192 MB ceiling meant that *every single model* in the benchmark received $S_{\text{mem}} = 1.0$, completely flattening memory differentiation between 1.57 GB (Alkd) and 6.63 GB (Qwen3 BF16).
3. **Metric Collapse**: For all fast edge models, the formula collapsed into $0.70 + 0.30 \times S_{\text{acc}}$, penalizing models with exceptional speed or ultra-low memory footprints.

## Decision
We report **both** scoring metrics side-by-side in all documentation and exports:
1. **Official Composite Score**: Preserved for historical compatibility and continuity with upstream reports.
2. **Continuous Audited Score**: Linear unclipped min-max normalization across the active evaluation spectrum:
   $$\hat{S}_{\text{speed}} = \frac{\text{Speed} - \text{Speed}_{\min}}{\text{Speed}_{\max} - \text{Speed}_{\min}}$$
   $$\hat{S}_{\text{mem}} = \frac{\text{Mem}_{\max} - \text{Mem}}{\text{Mem}_{\max} - \text{Mem}_{\min}}$$
   $$\hat{S}_{\text{acc}} = \frac{\text{WER}_{\max} - \text{WER}}{\text{WER}_{\max} - \text{WER}_{\min}}$$
   $$\text{Continuous Score} = 0.40 \times \hat{S}_{\text{speed}} + 0.30 \times \hat{S}_{\text{mem}} + 0.30 \times \hat{S}_{\text{acc}}$$

## Consequences
- Continuous scoring correctly spotlights:
  - **#1 Overall: Parakeet-TDT-0.6B (0.882)** for its extraordinary $72.99\times$ throughput (126 ms P50 latency).
  - **#2 Overall: Alkd Qwen3 audio8/text4 (0.662)** for its world-class memory frugality (1,573 MB peak VRAM) paired with 0.0242 WER.
- Both columns are exposed in `outputs/ALL_RESULTS.csv` and documented in detail in `README.md` and `README_zh.md`.
