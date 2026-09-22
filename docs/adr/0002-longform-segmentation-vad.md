# ADR 0002: Default Long-form Segmentation Strategy (Silero VAD vs RMS Energy)

## Status
Accepted

## Context
Modern speech foundation models have disparate input length limits:
- Whisper utilizes a native 30-second sliding-window architecture capable of processing hour-long continuous streams without manual chunking.
- Models like Qwen3-ASR and VibeVoice-ASR are constrained by positional embedding caps or upstream length limits (Qwen3 measured cap $\le 300\text{s}$, VibeVoice upstream code has a silent 59-minute clip).

To evaluate these models on our hour-scale continuous audio benchmark (~62.2 min / 3,731.8s), automated segmentation into sub-60s chunks is required. We compared two distinct chunking engines:
1. **RMS Energy Silence Detection**: Splits when audio signal falls below $-45\text{ dB}$ for at least $250\text{ ms}$ within a search lookback window.
2. **Silero Voice Activity Detection (VAD)**: Deep learning voice-activity classifier detecting speech onset and offsets, splitting at speech pauses with natural padding.

## Decision
We adopt **Silero VAD (60s target)** as the default segmentation engine across the repository:
1. **Memory Reduction**: On the identical continuous benchmark audio, Silero VAD (82 segments) lowered Qwen3-INT8 peak memory consumption by **14%** (from 4,767 MB down to 4,082 MB).
2. **Boundary Quality**: RMS energy silence detection risks splitting mid-word during soft phonemes or noisy pauses. Silero VAD acoustic probabilities ensure cuts occur strictly during true inter-utterance pauses.
3. **Rejection of 300s Chunking**: In empirical tests (`VAD-300s`), scaling chunk lengths to 300s resulted in:
   - 3.4x slower processing (RTF 0.4871 vs 0.1428)
   - 1.7x higher peak RAM (8,035 MB vs 4,082 MB)
   - Zero accuracy improvement.
   Therefore, 60s target chunking is enforced as the optimal operational boundary.

## Consequences
- `configs/benchmark.json` configures `manifests/segments_longform_vad60s.json` as the default segmentation manifest.
- All benchmark tables, scripts, and exports must explicitly label the segmentation methodology (`Silero VAD (60s)` vs `RMS Energy (60s)` vs `None (Native Continuous)`).
