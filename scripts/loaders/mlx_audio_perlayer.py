"""Loader shim for per-layer quantized MLX checkpoints (e.g. Alkd audio8/text4).

Standard mlx_audio uses `apply_quantization` which first consults the model's
`model_quant_predicate`. In Qwen3ASRModel, this predicate hardcodes
`not p.startswith("audio_tower")`, which vetoes the 147 audio_tower 8-bit layers
before consulting the checkpoint's per-layer config. This causes 294 parameters
to fail loading with Linear vs QuantizedLinear mismatch errors.

This shim temporarily patches `Qwen3ASRModel.model_quant_predicate` to a staticmethod
returning True during model loading, allowing `apply_quantization` to defer to
the checkpoint's own per-layer quantization specification without modifying
the locked site-packages.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Union
import mlx.core as mx
import mlx.nn as nn
from mlx_audio.stt.utils import load_model


@contextmanager
def patch_qwen3_quant_predicate():
    """Temporarily patch inner Qwen3ASRModel.model_quant_predicate."""
    import mlx_audio.stt.models.qwen3_asr.qwen3_asr as qwen_module
    
    # Locate inner model class
    inner_cls = getattr(qwen_module, "Qwen3ASRModel", None)
    if inner_cls is None:
        yield
        return

    orig_predicate = getattr(inner_cls, "model_quant_predicate", None)
    try:
        # Must be staticmethod so apply_quantization calls predicate(path, module) without self
        inner_cls.model_quant_predicate = staticmethod(lambda p, m: True)
        yield
    finally:
        if orig_predicate is not None:
            inner_cls.model_quant_predicate = orig_predicate


def load_qwen3_asr_per_layer(
    model_path: Union[str, Path],
    strict: bool = True,
    **kwargs
) -> nn.Module:
    """Load a per-layer quantized Qwen3-ASR model with predicate shim."""
    with patch_qwen3_quant_predicate():
        model = load_model(str(model_path), model_type="qwen3_asr", strict=strict, **kwargs)
    return model
