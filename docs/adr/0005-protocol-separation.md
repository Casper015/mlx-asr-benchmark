# ADR 0005: Strict Protocol Separation and Ground Truth Attribution

## Status
Accepted

## Context
ASR evaluations in the industry often conflate incompatible test setups:
1. Comparing continuous unsegmented streams against short sliced audio segments.
2. Mixing human-annotated phonetically normalized ground truths with machine-generated auto-captions.
3. Comparing diarized speaker-prefixed outputs against raw conversational turns.

Such practices introduce substantial statistical artifacts and lead to false conclusions.

## Decision
We enforce three non-negotiable benchmark disciplines:
1. **Never Merge Protocols into a Single Ranking Table**:
   - `native_continuous` (e.g. Whisper Turbo, TEA) and `auto_segmented` (e.g. Qwen3 INT8, VibeVoice 7B) must be published in distinct tables. A sliced pipeline incurs chunking, stitching, and boundary overhead that native sliding-window engines do not experience.
2. **Strict Ground Truth Provenance (Human Verified vs LOOSE_REF)**:
   - On the 1,000 Pipecat conversational benchmark, the reference is human-verified audio turns (`smart-turn-data-v3.1-train`), normalized strictly by rule: lowercasing, contraction expansion, punctuation removal, and filler word elimination (`pipecat-ai/stt-benchmark` methodology).
   - On unannotated long-form audio where machine subtitles are used as reference, the resulting metric MUST be designated as `LOOSE_REF` and cannot be reported as absolute WER.
3. **Dual-Criteria Reporting for Structured Outputs**:
   - VibeVoice outputs speaker tags (e.g. `Speaker 1:`). Raw normalized WER is **0.1470** (due to spurious speaker tokens). Stripping speaker prefixes reveals true acoustic WER of **0.0679**. Both numbers must be reported together.
   - For defective community ports (e.g. `aufklarer 4-bit`), we explicitly document the failure mode (186/1000 template failures outputting literal `language` token) while clarifying that the upstream `mlx-community 4-bit` model is completely healthy (0.0241 WER).

## Consequences
- Every report, table, and data export explicitly denotes the protocol and ground truth quality tier.
- Eliminates confirmation bias and misleading marketing comparisons.
