"""Qwen3-ASR Adapter for ASR Benchmark.

Runtime: mlx_speech==0.5.2 (locked, do not upgrade) or mlx_audio==0.5.4.
Uses the public mlx_speech.asr.load() entry point with a pinned HF revision,
or mlx_audio.stt.utils.load_model() for mlx-community checkpoints.

Reports ru_maxrss peak and MLX peak separately, never summed.
Language is auto-detected or optionally specified.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

QWEN3_MODEL_ID = "appautomaton/qwen3-asr-1.7b-int8-mlx"
QWEN3_REVISION = "31a02d835ce3b206e6d1e49bdbf33f16352a3f44"
QWEN3_MAX_NEW_TOKENS_CAP = 8192


def size_max_new_tokens(duration_sec: float) -> int:
    """Generous, duration-scaled generation budget.

    Floor 512 (~10x headroom over ~3.25 tok/s of English speech for <=60s
    inputs; 60s needs ~195 tokens). Cap 8192.
    """
    return max(512, min(QWEN3_MAX_NEW_TOKENS_CAP, int(duration_sec * 8)))


class Qwen3Adapter:
    def __init__(
        self,
        model_path: Optional[str] = None,
        precision: str = "int8",
        runtime: str = "mlx_speech"
    ):
        self.model_path = model_path or QWEN3_MODEL_ID
        self.precision = precision
        self.runtime = runtime
        self.model = None
        self._mlx_audio = (runtime == "mlx_audio")
        self._mlx_audio_perlayer = (runtime == "mlx_audio_perlayer")
        self.load_time_sec = 0.0

    def load(self) -> float:
        import mlx.core as mx
        start = time.perf_counter()
        if self._mlx_audio_perlayer:
            # Checkpoints that declare a per-layer quantization spec (e.g. audio
            # 8-bit + text 4-bit). The stock loader vetoes audio_tower before
            # consulting the spec, so a shim loader is required. See
            # scripts/loaders/mlx_audio_perlayer.py.
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from loaders.mlx_audio_perlayer import load_qwen3_asr_per_layer
            self.model = load_qwen3_asr_per_layer(self.model_path, strict=True)
            mx.eval(self.model.parameters())
        elif self._mlx_audio:
            from mlx_audio.stt.utils import load_model
            self.model = load_model(self.model_path, model_type="qwen3_asr", strict=True)
            mx.eval(self.model.parameters())
        else:
            import mlx_speech
            model_id = self.model_path or QWEN3_MODEL_ID
            self.model = mlx_speech.asr.load(model_id, revision=QWEN3_REVISION)
            try:
                params = self.model._runtime.model.parameters()
                mx.eval(params)
            except AttributeError:
                pass
        self.load_time_sec = time.perf_counter() - start
        return self.load_time_sec

    def _transcribe(self, audio_path: str, max_new_tokens: int) -> Dict[str, Any]:
        import mlx.core as mx
        if self.model is None:
            raise RuntimeError("Call load() first")
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        start = time.perf_counter()
        if self._mlx_audio or self._mlx_audio_perlayer:
            result = self.model.generate(str(audio_path), max_tokens=int(max_new_tokens), verbose=False)
            mx.synchronize()
            text = str(getattr(result, "text", "") or "").strip()
            language = str(getattr(result, "language", "unknown"))
            tokens = list(getattr(result, "tokens", None) or [])
            prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
            tokens_available = bool(tokens)
            if not tokens_available:
                generation_tokens = int(getattr(result, "generation_tokens", 0) or 0)
                tokens_available = generation_tokens > 0
            else:
                generation_tokens = len(tokens)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mlx_peak = mx.get_peak_memory() / (1024 ** 2) if hasattr(mx, "get_peak_memory") else 0.0
            return {
                "text": text,
                "segments": [],
                "elapsed_sec": time.perf_counter() - start,
                "peak_rss_mb": peak / (1024 ** 2 if sys.platform == "darwin" else 1024),
                "mlx_peak_memory_mb": mlx_peak,
                "generation_tokens": generation_tokens,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": int(max_new_tokens),
                "tokens_available": tokens_available,
                "language": language,
                "language_enforced": False,
                "token_limit_reached": bool(generation_tokens) and generation_tokens >= int(max_new_tokens),
            }

        runtime = getattr(self.model, "_runtime", None)
        if runtime is not None and hasattr(runtime, "transcribe"):
            result = runtime.transcribe(
                str(audio_path),
                sample_rate=16000,
                language=None,
                max_new_tokens=int(max_new_tokens),
            )
            text = str(getattr(result, "text", "") or "").strip()
            language = str(getattr(result, "language", "unknown"))
            tokens = list(getattr(result, "tokens", None) or [])
            prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
            tokens_available = True
        else:
            out = self.model.generate(str(audio_path))
            mx.synchronize()
            text = str(getattr(out, "text", "") or "").strip()
            language = str(getattr(out, "language", "unknown"))
            tokens, prompt_tokens = [], 0
            tokens_available = False

        mx.synchronize()
        elapsed = time.perf_counter() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mlx_peak = mx.get_peak_memory() / (1024 ** 2) if hasattr(mx, "get_peak_memory") else 0.0
        generation_tokens = len(tokens)
        return {
            "text": text,
            "segments": [],
            "elapsed_sec": elapsed,
            "peak_rss_mb": peak / (1024 ** 2 if sys.platform == "darwin" else 1024),
            "mlx_peak_memory_mb": mlx_peak,
            "generation_tokens": generation_tokens,
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": int(max_new_tokens),
            "tokens_available": tokens_available,
            "language": language,
            "language_enforced": False,
            "token_limit_reached": tokens_available and generation_tokens >= int(max_new_tokens),
        }

    def transcribe_segment(
        self,
        audio_path: str,
        language: str = "en",
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        import soundfile as sf
        duration = float(sf.info(str(audio_path)).duration)
        budget = int(max_new_tokens) if max_new_tokens else size_max_new_tokens(duration)
        return self._transcribe(audio_path, budget)

    def transcribe_longform(
        self,
        audio_path: str,
        language: str = "en",
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        return self.transcribe_segment(audio_path, language, temperature, max_new_tokens)

    def cleanup(self):
        self.model = None
        gc.collect()
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except ImportError:
            pass
