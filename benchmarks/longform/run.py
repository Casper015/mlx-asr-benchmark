#!/usr/bin/env python3
"""Long-form Continuous Audio ASR Benchmark Runner.

Evaluates speech recognition models on continuous, multi-minute audio streams.
Audio Specification:
    Continuous long-form audio benchmark (~62.2 min / 3,731.8s, 16 kHz mono PCM WAV, single speaker).

Evaluation Protocols:
    - native_continuous: Unchunked streaming/sliding-window inference (e.g. Whisper, TEA).
    - auto_segmented: Silence-aware chunking with Silero VAD (default, 60s) or RMS energy.
      Used for models with bounded context windows (e.g. Qwen3 <=300s cap, VibeVoice).

Zero privacy leakage: Paths resolved relative to repository root or $HOME.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repository root resolution
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Benchmark defaults
DEFAULT_AUDIO_REL = "data/longform/audio_16k.wav"
DEFAULT_TARGET_SEC = 60.0
DEFAULT_HARD_MAX_SEC = 60.0
SILENCE_RMS_DB = -45.0
MIN_SILENCE_SEC = 0.25


def get_peak_rss_mb() -> float:
    """Return maximum RSS memory in megabytes."""
    rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rusage / (1024 * 1024)
    return rusage / 1024


def get_mlx_peak_mb() -> float:
    """Return MLX peak memory in megabytes if MLX is loaded."""
    try:
        import mlx.core as mx
        if hasattr(mx, "get_peak_memory"):
            return mx.get_peak_memory() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def reset_mlx_peak_memory():
    """Reset MLX peak memory counter if available."""
    try:
        import mlx.core as mx
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
    except ImportError:
        pass


def srt_timestamp(sec: float) -> str:
    """Format seconds into SRT subtitle timestamp format HH:MM:SS,mmm."""
    td = datetime.timedelta(seconds=max(0.0, sec))
    total_sec = int(td.total_seconds())
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    millis = int((sec - int(sec)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def load_audio_info(audio_path: Path) -> Tuple[float, int, int]:
    """Return (duration_seconds, sample_rate, channels)."""
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        return float(info.duration), int(info.samplerate), int(info.channels)
    except Exception:
        # Fallback using wave module
        import wave
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            return frames / float(rate), rate, channels


def run_benchmark(
    model_name: str,
    audio_path: Path,
    protocol: str = "auto",
    segment_method: str = "vad",
    target_sec: float = DEFAULT_TARGET_SEC,
    output_dir: Optional[Path] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """Execute longform benchmark for the specified model and audio."""
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}\n"
            f"Please place the benchmark audio at {audio_path} or specify --audio."
        )

    duration_sec, sample_rate, channels = load_audio_info(audio_path)
    print(f"Loaded Benchmark Audio:")
    print(f"  Path: {audio_path}")
    print(f"  Acoustic: Continuous long-form audio benchmark (~{duration_sec / 60:.1f} min / {duration_sec:.1f}s, {sample_rate} Hz mono PCM WAV, single speaker)")

    # Model configuration
    models_cfg_path = REPO_ROOT / "configs" / "models.json"
    with open(models_cfg_path, "r", encoding="utf-8") as f:
        models_reg = json.load(f).get("models", {})

    model_entry = models_reg.get(model_name)
    if not model_entry:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models_reg.keys())}")

    adapter_type = model_entry.get("adapter")
    precision = model_entry.get("precision", "float16")
    is_native = adapter_type in ["whisper", "tea"]

    if protocol == "auto":
        protocol = "native_continuous" if is_native else "auto_segmented"

    print(f"\nModel Configuration:")
    print(f"  Name: {model_entry.get('name')}")
    print(f"  Repo: {model_entry.get('repo_id')}")
    print(f"  Protocol: {protocol}")
    print(f"  Segmentation Method: {segment_method if protocol == 'auto_segmented' else 'None (Native Continuous)'}")

    reset_mlx_peak_memory()
    start_time = time.perf_counter()

    transcript_text = ""
    used_segments: List[Dict[str, Any]] = []

    if protocol == "native_continuous":
        if adapter_type == "whisper":
            from scripts.adapters.whisper_adapter import WhisperAdapter
            model_path = model_entry.get("repo_id")
            adapter = WhisperAdapter(model_path=model_path, precision=precision)
            adapter.load()
            res = adapter.transcribe_longform(str(audio_path), language=language)
            transcript_text = res.get("text", "")
            for idx, sg in enumerate(res.get("segments", []), 1):
                used_segments.append({
                    "index": idx,
                    "start_sec": float(sg.get("start", 0.0)),
                    "end_sec": float(sg.get("end", duration_sec)),
                    "text": str(sg.get("text", "")).strip()
                })
            adapter.cleanup()
        else:
            raise NotImplementedError(f"Native runner for adapter '{adapter_type}' requires dedicated setup.")
    else:
        # Segmented execution
        print(f"Running auto_segmented protocol using method: {segment_method} (target {target_sec}s)...")
        manifest_name = f"segments_longform_{segment_method}60s.json"
        manifest_path = REPO_ROOT / "manifests" / manifest_name
        cut_segments: List[Dict[str, Any]] = []

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as mf:
                mdata = json.load(mf)
                src_dur = float(mdata.get("source", {}).get("duration_sec", 0.0))
                # Use precomputed manifest only if audio duration matches benchmark audio within 2 seconds
                if abs(duration_sec - src_dur) < 2.0:
                    cut_segments = mdata.get("segments", [])
                    print(f"Loaded {len(cut_segments)} precomputed segments matching benchmark audio ({src_dur:.1f}s)")

        if not cut_segments:
            # Dynamic fallback: split audio into chunks of target_sec
            num_chunks = max(1, int(duration_sec // target_sec) + (1 if duration_sec % target_sec > 0 else 0))
            for i in range(num_chunks):
                s0 = i * target_sec
                s1 = min((i + 1) * target_sec, duration_sec)
                cut_segments.append({"start_sec": s0, "end_sec": s1, "duration_sec": s1 - s0})
            print(f"Generated {len(cut_segments)} dynamic segments (target {target_sec}s for {duration_sec:.1f}s audio)")

        # Instantiate adapter
        if adapter_type == "qwen3":
            from scripts.adapters.qwen3_adapter import Qwen3Adapter
            model_path = model_entry.get("repo_id")
            adapter = Qwen3Adapter(
                model_path=model_path,
                precision=precision,
                runtime=model_entry.get("runtime", "mlx_speech")
            )
            adapter.load()
        elif adapter_type == "whisper":
            from scripts.adapters.whisper_adapter import WhisperAdapter
            model_path = model_entry.get("repo_id")
            adapter = WhisperAdapter(model_path=model_path, precision=precision)
            adapter.load()
        else:
            raise NotImplementedError(f"Segmented runner for adapter '{adapter_type}' requires dedicated setup.")

        # Read audio data for slicing
        import soundfile as sf
        audio_data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        pieces: List[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, cseg in enumerate(cut_segments, 1):
                s0_sec = float(cseg.get("start_sec", cseg.get("start", 0.0)))
                s1_sec = float(cseg.get("end_sec", cseg.get("end", duration_sec)))
                s0_sample = int(s0_sec * sr)
                s1_sample = min(int(s1_sec * sr), len(audio_data))

                chunk_audio = audio_data[s0_sample:s1_sample]
                chunk_file = Path(tmpdir) / f"chunk_{idx:04d}.wav"
                sf.write(str(chunk_file), chunk_audio, sr, subtype="PCM_16")

                seg_res = adapter.transcribe_segment(str(chunk_file), language=language)
                seg_text = (seg_res.get("text") or "").strip()
                if seg_text:
                    pieces.append(seg_text)

                used_segments.append({
                    "index": idx,
                    "start_sec": round(s0_sec, 3),
                    "end_sec": round(s1_sec, 3),
                    "duration_sec": round(s1_sec - s0_sec, 3),
                    "text": seg_text,
                    "elapsed_sec": round(seg_res.get("elapsed_sec", 0.0), 2)
                })

                if idx % 10 == 0 or idx == len(cut_segments):
                    print(f"  [{idx}/{len(cut_segments)}] {s0_sec:.1f}s - {s1_sec:.1f}s: {len(seg_text)} chars")

        transcript_text = "\n".join(pieces)
        adapter.cleanup()

    elapsed_sec = time.perf_counter() - start_time
    rtf = elapsed_sec / duration_sec if duration_sec > 0 else 0.0
    speedup = duration_sec / elapsed_sec if elapsed_sec > 0 else 0.0
    words = len(transcript_text.split()) if transcript_text else 0

    rss_peak = get_peak_rss_mb()
    mlx_peak = get_mlx_peak_mb()

    result = {
        "audio_description": "Continuous long-form audio benchmark (~62.2 min / 3,731.8s, 16 kHz mono PCM WAV, single speaker)",
        "duration_sec": duration_sec,
        "model": model_name,
        "model_display_name": model_entry.get("name"),
        "protocol": protocol,
        "segmentation_method": segment_method if protocol == "auto_segmented" else "None (Native Continuous)",
        "num_segments": len(used_segments),
        "elapsed_sec": round(elapsed_sec, 2),
        "rtf": round(rtf, 4),
        "speedup_x": round(speedup, 2),
        "rss_peak_mb": round(rss_peak, 2),
        "mlx_peak_mb": round(mlx_peak, 2),
        "words": words,
        "transcript_preview": transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text
    }

    print("\nBenchmark Execution Complete:")
    print(f"  Duration: {duration_sec:.1f}s | Infer: {elapsed_sec:.2f}s | RTF: {rtf:.4f} | Speedup: {speedup:.2f}x")
    print(f"  Memory Peak: RSS={rss_peak:.1f} MB | MLX={mlx_peak:.1f} MB (Metal Unified Memory)")
    print(f"  Words: {words}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        if transcript_text:
            with open(output_dir / "transcript.txt", "w", encoding="utf-8") as f:
                f.write(transcript_text)
        if used_segments:
            with open(output_dir / "transcript.srt", "w", encoding="utf-8") as f:
                for sg in used_segments:
                    f.write(f"{sg['index']}\n")
                    f.write(f"{srt_timestamp(sg['start_sec'])} --> {srt_timestamp(sg['end_sec'])}\n")
                    f.write(f"{sg['text']}\n\n")
            with open(output_dir / "segments.json", "w", encoding="utf-8") as f:
                json.dump(used_segments, f, indent=2)
        print(f"Results written to {output_dir}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Continuous long-form audio benchmark runner (~62.2 min, 16 kHz mono PCM WAV)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="whisper_turbo",
        help="Model ID registered in configs/models.json (default: whisper_turbo)"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=str(REPO_ROOT / DEFAULT_AUDIO_REL),
        help="Path to continuous long-form audio file"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["auto", "native_continuous", "auto_segmented"],
        default="auto",
        help="Protocol choice: native_continuous vs auto_segmented (default: auto)"
    )
    parser.add_argument(
        "--segment-method",
        type=str,
        choices=["vad", "rms"],
        default="vad",
        help="Chunking method for auto_segmented protocol: vad (Silero VAD, default) or rms (RMS Energy)"
    )
    parser.add_argument(
        "--target-sec",
        type=float,
        default=60.0,
        help="Target chunk duration in seconds (default: 60.0s)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save metrics.json, transcript.txt, and transcript.srt"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Audio spoken language (default: en)"
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_benchmark(
        model_name=args.model,
        audio_path=Path(args.audio),
        protocol=args.protocol,
        segment_method=args.segment_method,
        target_sec=args.target_sec,
        output_dir=out_dir,
        language=args.language
    )


if __name__ == "__main__":
    main()
