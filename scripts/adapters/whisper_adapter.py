"""Whisper Adapter for ASR Benchmark.

Uses mlx_whisper with local model snapshots or Hugging Face repos.
Supports native long-form continuous audio and conversational turn transcription,
with explicit memory management and garbage collection.
"""

import gc
import resource
import sys
import time
from pathlib import Path
import mlx.core as mx
from mlx_whisper.transcribe import ModelHolder, transcribe


class WhisperAdapter:
    def __init__(self, model_path: str, precision: str = "float16"):
        self.model_path = model_path
        self.precision = precision
        self.model = None
        self.load_time_sec = 0.0

    def load(self):
        start = time.perf_counter()
        dtype = mx.float16 if self.precision == "float16" else mx.float32
        self.model = ModelHolder.get_model(self.model_path, dtype=dtype)
        # Force evaluation to ensure weights are fully in memory
        mx.eval(self.model.parameters())
        self.load_time_sec = time.perf_counter() - start
        return self.load_time_sec

    def transcribe_longform(
        self,
        audio_path: str,
        language: str = "en",
        temperature: float = 0.0,
        condition_on_previous_text: bool = True
    ) -> dict:
        """Transcribe continuous long-form audio natively using sliding-window decoding."""
        start = time.perf_counter()
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()

        result = transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_path,
            language=language,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            verbose=False,
            fp16=(self.precision == "float16")
        )
        elapsed = time.perf_counter() - start

        # Peak RSS via getrusage; MLX peak recorded separately, never summed.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = peak / (1024 ** 2 if sys.platform == 'darwin' else 1024)
        mlx_peak_mb = (mx.get_peak_memory() / (1024 * 1024)) if hasattr(mx, "get_peak_memory") else 0.0

        return {
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", []),
            "elapsed_sec": elapsed,
            "peak_rss_mb": peak_rss_mb,
            "mlx_peak_memory_mb": mlx_peak_mb
        }

    def transcribe_segment(self, audio_path: str, language: str = "en", temperature: float = 0.0) -> dict:
        """Transcribe a short segment (or single conversational turn)."""
        start = time.perf_counter()
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()

        result = transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_path,
            language=language,
            temperature=temperature,
            condition_on_previous_text=False,
            verbose=False,
            fp16=(self.precision == "float16")
        )
        elapsed = time.perf_counter() - start

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = peak / (1024 ** 2 if sys.platform == 'darwin' else 1024)
        mlx_peak_mb = (mx.get_peak_memory() / (1024 * 1024)) if hasattr(mx, "get_peak_memory") else 0.0

        return {
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", []),
            "elapsed_sec": elapsed,
            "peak_rss_mb": peak_rss_mb,
            "mlx_peak_memory_mb": mlx_peak_mb
        }

    def cleanup(self):
        """Strict memory cleanup and cache release."""
        self.model = None
        if hasattr(ModelHolder, "model"):
            ModelHolder.model = None
        if hasattr(ModelHolder, "model_path"):
            ModelHolder.model_path = None
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
