# ADR 0001: Zero-Cost Quantization Frontier in Modern Edge ASR

## Status
Accepted

## Context
When deploying Automatic Speech Recognition (ASR) models to edge devices (e.g. Apple Silicon M-series Macs), developers frequently face the decision of whether to deploy full-precision weights (BF16/FP16) or quantized checkpoints (INT8, 8-bit, 6-bit, 5-bit, 4-bit). In conventional natural language processing (NLP), aggressive low-bit quantization (such as 4-bit or 5-bit) typically incurs a measurable degradation in perplexity or task accuracy.

In this benchmark, we rigorously tested 19 conditions across 1,000 real-world conversational turns (Pipecat smart-turn dataset) and an hour-scale continuous audio stream.

## Decision
We established that within the Qwen3-ASR architecture on Apple Silicon MLX:
1. **Identical Normalized Word Error Rate across 8-bit, 6-bit, and 5-bit**:
   - `Qwen3-ASR-1.7B 8-bit (mc)`: Normalized WER = **0.0217**, Perfect = 70.9%
   - `Qwen3-ASR-1.7B 6-bit (mc)`: Normalized WER = **0.0217**, Perfect = 71.0%
   - `Qwen3-ASR-1.7B 5-bit (mc)`: Normalized WER = **0.0217**, Perfect = 70.8%
   - `Qwen3-ASR-1.7B INT8 (appautomaton)`: Normalized WER = **0.0218**, Perfect = 71.0%
   - `Qwen3-ASR-1.7B BF16 (mc)`: Normalized WER = **0.0221**, Perfect = 70.5%
2. **BF16 is NOT more accurate than quantized variants**:
   The full-precision BF16 model scored 0.0221, which is statistically indistinguishable from or slightly worse than the 0.0217 of 5/6/8-bit models (differences of 0.0004 fall well within reference annotation noise).
3. **The Trade-Off Axis is Latency and Memory, Not Accuracy**:
   - Moving from 8-bit to 5-bit increases speed from **10.8x to 12.7x** (+17.6%) while shrinking peak Metal Unified Memory from **3,272 MB to 2,657 MB** (-18.8%).
   - Stepping down to 4-bit (`mlx-community 4-bit`) increases throughput to **17.8x** (P50 latency drops to **557 ms**) with peak memory dropping to **2,452 MB**, at the cost of only a minor WER increase (from 0.0217 to 0.0241).

## Consequences
- Benchmark documentation and guidelines must explicitly state: **"Do not claim 8-bit or BF16 is 'more accurate' than 5-bit or 6-bit."**
- The selection between 4-bit, 5-bit, 6-bit, and 8-bit is strictly a hardware budget optimization (allocating available VRAM and latency constraints).
- For edge voice agents requiring sub-600ms turn turnaround, `Qwen3-ASR-1.7B 4-bit (mc)` represents the ideal Pareto sweet spot (557 ms P50, 17.8x speedup).
