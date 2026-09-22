# Project Context & Engineering Architecture

## 1. Overview and Core Purpose
This repository delivers a rigorous, production-grade Automatic Speech Recognition (ASR) benchmark and local transcription toolkit specifically optimized for Apple Silicon via Apple's MLX machine learning framework.

The project evaluates 19 model configurations across two primary operational domains:
1. **Low-Latency Conversational STT (1,000 Turns)**: Sourced from `pipecat-ai/smart-turn-data-v3.1-train`, testing real-world conversational turns for voice agents.
2. **Continuous Long-Form Audio Benchmark (~62.2 min / 3,731.8s)**: Single speaker, 16 kHz mono PCM WAV, evaluating throughput, memory footprint, and long-horizon stability.

---

## 2. Hardware and Environment Specification
All benchmark figures in this repository were empirically measured under locked conditions:
- **Processor**: Apple Silicon M5 (8 GPU cores)
- **Memory Architecture**: 24 GB Unified Memory (Metal architecture)
- **Operating System**: macOS (Darwin)
- **Core Runtime**: Python 3.14 / Apple MLX 0.32.2
- **Memory Ceiling**: 20 GB hard ceiling (exceeding either MLX peak or process RSS constitutes failure)

---

## 3. Core Technical Conclusions

### 1. The Zero-Cost Quantization Frontier
Contrary to conventional NLP assumptions where sub-8-bit quantization harms accuracy:
- Qwen3-ASR-1.7B exhibits **identical 0.0217 Normalized WER across 8-bit, 6-bit, and 5-bit**.
- Full-precision BF16 achieves 0.0221 Normalized WER (statistically identical or marginally worse due to reference noise).
- The optimization trade-off is strictly **speed vs memory**:
  - 8-bit: 10.8x speedup, 3,272 MB peak Metal VRAM.
  - 5-bit: 12.7x speedup, 2,657 MB peak Metal VRAM.
  - 4-bit (`mlx-community`): 17.8x speedup, 2,452 MB peak Metal VRAM, 557 ms P50 latency.

### 2. Runtime Path Dictates Throughput
Identical weights and precision run at radically different speeds depending on runtime dispatch:
- Under `mlx_audio`: Qwen3 BF16 runs at P50 = 1,588 ms (6.6x).
- Under `mlx_speech`: Qwen3 BF16 runs at P50 = 6,085 ms (1.9x).
This 3.8x disparity stems entirely from runtime implementation details rather than model architecture.

### 3. Dual-Scoring Paradigm
- **Official Composite Formula**: Saturated on speed ($\ge 15\times \rightarrow 1.0$) and memory ($\le 8,192\text{ MB} \rightarrow 1.0$).
- **Continuous Audited Formula**: Normalizes continuously across the real performance spectrum:
  - Highlights **Parakeet-TDT-0.6B (#1)** for unprecedented 72.99x speedup (126 ms P50).
  - Highlights **Alkd Qwen3 audio8/text4 (#2)** for lowest peak VRAM (1,573 MB) with 0.0242 WER.

### 4. Protocol Isolation
- `native_continuous`: Unchunked streaming / sliding window (Whisper, TEA).
- `auto_segmented`: Silence-aware chunking (Silero VAD 60s default; RMS energy).
These two protocols are never ranked in a single combined table.

---

## 4. Repository File Structure
```
github_export/
├── README.md                      # English showcase & primary benchmark documentation
├── README_zh.md                   # Full Chinese showcase & technical post-mortem
├── CONTEXT.md                     # Engineering context & design specifications
├── LICENSE                        # MIT License
├── setup.sh                       # Environment bootstrap script
├── requirements.txt               # Dependency specification
├── assets/                        # 8 publication-grade high-DPI benchmark figures
├── benchmarks/
│   └── longform/                  # Anonymized continuous long-form audio benchmark
│       ├── README.md              # Longform acoustic protocol specification
│       └── run.py                 # Standalone longform benchmark CLI
├── configs/
│   ├── benchmark.json             # Global benchmark parameters & scope definitions
│   └── models.json                # Complete model registry with Hugging Face URLs
├── docs/
│   └── adr/                       # Architectural Decision Records (0001 to 0005)
├── manifests/
│   ├── audio_manifest.json        # Audio metadata & dataset provenance
│   ├── models_manifest.json       # SHA-256 integrity hashes for model weights
│   ├── segments_longform_rms60s.json   # 68 RMS energy cutpoints
│   └── segments_longform_vad60s.json   # 82 Silero VAD cutpoints (Default)
├── outputs/
│   ├── ALL_DATA.json              # Unified machine-readable consolidated data
│   ├── ALL_RESULTS.csv            # 19-condition conversational benchmark table
│   └── HOUR_RESULTS.csv           # Continuous hour-scale benchmark results
├── scripts/
│   ├── adapters/                  # Model adapters (Whisper, Qwen3, etc.)
│   ├── loaders/                   # Dynamic runtime loader shims
│   └── generate_showcase_charts.py# Matplotlib publication chart generator
└── tests/                         # Full automated test suite (invariants, privacy, scoring)
```
