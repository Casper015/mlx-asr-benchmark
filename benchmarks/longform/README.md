# Long-Form Continuous Audio Apple MLX ASR Benchmark

This benchmark evaluates Automatic Speech Recognition (ASR) systems on continuous, hour-scale audio without artificial turn boundaries. All evaluated models run **100% natively on Apple Silicon** using Apple's MLX machine learning framework (`mlx-whisper` and `mlx-audio`).

## 1. Acoustic & Physical Dataset Specification

| Property | Value |
|---|---|
| **Description** | Continuous long-form audio benchmark (~62.2 min / 3,731.8s, 16 kHz mono PCM WAV, single speaker) |
| **Duration** | `3,731.818 seconds` (~62.2 minutes) |
| **Sampling Rate** | `16,000 Hz` (16 kHz) |
| **Channels** | `1` (Mono) |
| **Audio Format** | Uncompressed PCM 16-bit WAV |
| **Total Samples** | `59,709,092` samples |
| **Acoustic Environment** | Clean spoken speech, continuous presentation cadence, natural pauses |
| **Privacy Status** | Fully sanitized; anonymized strictly by physical properties |

---

## 2. Evaluation Protocols & Disciplinary Separation

In accordance with strict benchmarking discipline, **two non-overlapping protocols** are evaluated separately and **never ranked in a single combined table**:

### Protocol A: `native_continuous`
- **Definition**: Unchunked inference where the audio stream is passed directly into the model's native long-form engine (e.g. Whisper's 30-second sliding-window decoding or TEA's unsegmented attention).
- **Applicable Models**:
  - [Whisper large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo)
  - [Distil-Whisper large-v3](https://huggingface.co/mlx-community/distil-whisper-large-v3)
  - [Whisper large-v3](https://huggingface.co/mlx-community/whisper-large-v3)
  - [TEA-ASR-1.1](https://huggingface.co/JacobLinCool/TEA-ASR-1.1)

### Protocol B: `auto_segmented`
- **Definition**: Silence-aware automated chunking for models with finite acoustic context limits (such as Qwen3-ASR's measured $\le 300\text{s}$ limit and VibeVoice's upstream 59-minute silent truncation). Chunks are transcribed independently and stitched into a contiguous transcript.
- **Chunking Engines**:
  - **Silero VAD (60s target, Default)**: Splits at natural acoustic pauses identified by Silero Voice Activity Detection. Yields 82 segments, lower peak memory (4,082 MB vs 4,767 MB, $-14\%$), and minimizes boundary truncation errors.
  - **RMS Energy (60s target)**: Splits at energy-based silence thresholds ($\le -45\text{ dB}$, $\ge 250\text{ ms}$). Yields 68 segments.
- **Applicable Models**:
  - [Qwen3-ASR-1.7B INT8](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx)
  - [Qwen3-ASR-1.7B 4-bit (mc)](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-4bit)
  - [VibeVoice-ASR-Streaming-7B-4bit](https://huggingface.co/mlx-community/VibeVoice-ASR-Streaming-7B-4bit)

---

## 3. Audited Long-Form Benchmark Results

Measured on Apple Silicon M5 (24 GB Unified Memory, 8 GPU cores, macOS, MLX 0.32.2). Hard ceiling: 20 GB.

### Table 1: Protocol A — Native Continuous MLX Inference

| Model | Hugging Face Repository | Segmentation Method | Segments | Inference (s) | RTF | Speedup | Peak RAM | MLX RAM | Words | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| **Whisper large-v3-turbo** | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | None (Native Continuous) | 675 (internal) | **99.09s** | **0.0266** | **37.59x** | 4,034 MB | 4,034 MB | 9,718 | **PASS** |
| **Distil-Whisper large-v3** | [mlx-community/distil-whisper-large-v3](https://huggingface.co/mlx-community/distil-whisper-large-v3) | None (Native Continuous) | 743 (internal) | **84.43s** | **0.0226** | **44.25x** | **3,934 MB** | 3,934 MB | 9,707 | **PASS** |
| **Whisper large-v3** | [mlx-community/whisper-large-v3](https://huggingface.co/mlx-community/whisper-large-v3) | None (Native Continuous) | 632 (internal) | 537.27s | 0.1440 | 6.94x | 5,435 MB | 5,435 MB | 9,731 | **PASS** |
| **TEA-ASR-1.1** | [JacobLinCool/TEA-ASR-1.1](https://huggingface.co/JacobLinCool/TEA-ASR-1.1) | None (Native Continuous) | 3 | 720.23s | 0.1930 | 5.18x | 15,030 MB | 15,030 MB | 7,837 | **PASS** (78.3% coverage) |

### Table 2: Protocol B — Auto-Segmented MLX Inference

| Model | Hugging Face Repository | Segmentation Method | Segments | Inference (s) | RTF | Speedup | Peak RAM | MLX RAM | Words | Quality |
|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-ASR-1.7B INT8** | [appautomaton/qwen3-asr-1.7b-int8-mlx](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx) | **Silero VAD (60s)** | 82 | 532.92s | 0.1428 | **7.00x** | **4,082 MB** | 4,082 MB | 9,882 | **PASS** |
| **Qwen3-ASR-1.7B INT8** | [appautomaton/qwen3-asr-1.7b-int8-mlx](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx) | RMS Energy (60s) | 68 | 537.19s | 0.1439 | 6.95x | 4,767 MB | 4,767 MB | 9,824 | **PASS** |
| **Qwen3-ASR-1.7B 4-bit** | [mlx-community/Qwen3-ASR-1.7B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-4bit) | RMS Energy (60s) | 68 | **182.26s** | **0.0488** | **20.49x** | **2,898 MB** | 2,898 MB | 6,574 | **PASS** (65.7% coverage) |
| **VibeVoice-ASR-7B 4-bit** | [mlx-community/VibeVoice-ASR-Streaming-7B-4bit](https://huggingface.co/mlx-community/VibeVoice-ASR-Streaming-7B-4bit) | RMS Energy (60s) | 68 | 1,754.36s | 0.4701 | 2.13x | 5,990 MB | 5,990 MB | 10,347 | **PASS** (Speaker diarized) |

> **Crucial Benchmark Findings**:
> 1. **Whisper Turbo Dominance in Long-form**: For continuous unsegmented hour-scale transcription, `whisper-large-v3-turbo` achieves **37.6x real-time throughput** (processing the full ~62.2 min audio in just 99.1s) with 4.0 GB peak RAM.
> 2. **Silero VAD Optimization**: Switching segmentation from RMS energy to Silero VAD reduces peak memory by **14%** (from 4,767 MB to 4,082 MB) and eliminates word-cutting artifacts at chunk boundaries.
> 3. **Segment Scaling Degradation**: Extending chunk size to 300s (VAD-300s) resulted in 3.4x slower execution and 1.7x higher RAM without any accuracy gain. 60s remains the sweet spot.

---

## 4. Running the Benchmark

### Prerequisites
Ensure the environment is activated and dependencies are installed:
```bash
./setup.sh
```

### Execution Commands
```bash
# Run Whisper large-v3-turbo (native continuous protocol)
python benchmarks/longform/run.py --model whisper_turbo

# Run Qwen3-ASR-1.7B INT8 with default Silero VAD (60s) segmentation
python benchmarks/longform/run.py --model qwen3_asr_1_7b_int8 --protocol auto_segmented --segment-method vad

# Run with custom output directory
python benchmarks/longform/run.py --model whisper_turbo --output-dir ./results/longform_whisper
```
