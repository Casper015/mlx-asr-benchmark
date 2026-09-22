"""Unit tests for runtime loader shims and adapters."""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestLoadersAndShims(unittest.TestCase):
    def test_patch_qwen3_quant_predicate_context(self):
        """Verify patch_qwen3_quant_predicate handles execution cleanly."""
        from scripts.loaders.mlx_audio_perlayer import patch_qwen3_quant_predicate

        # Even if mlx_audio is not installed or Qwen3ASRModel is not present,
        # the context manager should enter and exit gracefully.
        with patch_qwen3_quant_predicate():
            entered = True
        self.assertTrue(entered)

    def test_whisper_adapter_instantiation(self):
        """Verify WhisperAdapter module can be imported and initialized without errors."""
        from scripts.adapters.whisper_adapter import WhisperAdapter

        adapter = WhisperAdapter(model_path="dummy/path", precision="float16")
        self.assertEqual(adapter.precision, "float16")
        self.assertIsNone(adapter.model)

    def test_qwen3_adapter_instantiation(self):
        """Verify Qwen3Adapter module can be imported and initialized without errors."""
        from scripts.adapters.qwen3_adapter import Qwen3Adapter

        adapter = Qwen3Adapter(model_path="dummy/path", precision="int8", runtime="mlx_speech")
        self.assertEqual(adapter.precision, "int8")
        self.assertEqual(adapter.runtime, "mlx_speech")
        self.assertIsNone(adapter.model)


if __name__ == "__main__":
    unittest.main()
