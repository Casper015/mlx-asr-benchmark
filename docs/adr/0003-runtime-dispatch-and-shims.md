# ADR 0003: Runtime Dispatch Mechanics and Per-Layer Quantization Shims

## Status
Accepted

## Context
Running diverse MLX community model checkpoints exposed critical runtime divergences:

### 1. Runtime Path Divergence (mlx_audio vs mlx_speech)
For identical model architectures and precision (Qwen3-ASR-1.7B BF16):
- Checkpoints converted by `mlx-community` (e.g. `mlx-community/Qwen3-ASR-1.7B-bf16`) load via `mlx_audio` with a P50 latency of **1,588 ms** (6.6x speedup).
- Checkpoints converted by `appautomaton` (e.g. `appautomaton/qwen3-asr-1.7b-bf16-mlx`) require `mlx_speech` to align keys; running under `mlx_speech` produced a P50 latency of **6,085 ms** (1.9x speedup) — a **3.8x performance slowdown** purely driven by the inference engine runtime rather than model weights.

### 2. Per-Layer Quantized Checkpoints (Alkd audio8/text4)
The checkpoint `Alkd/Qwen3-ASR-1.7B-audio8-text4-mlx` applies heterogeneous quantization:
- Audio encoder (`audio_tower`): 147 layers quantized to 8-bit.
- Text decoder: quantized to 4-bit.

However, standard `mlx_audio` uses `apply_quantization()`, which queries `inner_cls.model_quant_predicate`. In the upstream `Qwen3ASRModel` definition, the predicate is hardcoded as `not p.startswith("audio_tower")`. Consequently, `mlx_audio` vetoed the audio encoder layers before reading the checkpoint's layer map, leading to 294 unquantized parameter mismatches and failure during weight loading.

## Decision
1. **Explicit Model Metadata Dispatch**:
   `configs/models.json` includes explicit `runtime` fields (`mlx_audio` vs `mlx_speech`) for every registered model, preventing runtime misconfiguration.
2. **Dynamic Context-Manager Predicate Shim (`mlx_audio_perlayer.py`)**:
   Instead of modifying or forking upstream site-packages, we implemented an in-memory monkey-patch shim:
   ```python
   @contextmanager
   def patch_qwen3_quant_predicate():
       inner_cls = getattr(qwen_module, "Qwen3ASRModel", None)
       orig = getattr(inner_cls, "model_quant_predicate", None)
       try:
           inner_cls.model_quant_predicate = staticmethod(lambda p, m: True)
           yield
       finally:
           inner_cls.model_quant_predicate = orig
   ```
   Note: The predicate must be wrapped in `staticmethod` so `apply_quantization(path, module)` does not receive an unintended `self` argument.

## Consequences
- The Alkd mixed-precision model runs flawlessly without patching site-packages, achieving **14.1x real-time speed** and setting the benchmark record for the **lowest peak VRAM consumption of all models: 1,573 MB**.
- Unit tests verify the shim's entry and clean exit under normal and exception conditions.
