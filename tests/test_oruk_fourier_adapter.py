"""Unit tests for Oruk Fourier 2B MLX subfolder config and adapter resolution."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from adapters.qwen3_adapter import Qwen3Adapter  # noqa: E402


class TestOrukFourierAdapter(unittest.TestCase):
    def test_models_json_has_oruk_fourier_2b(self):
        data = json.loads((REPO_ROOT / "configs" / "models.json").read_text(encoding="utf-8"))
        entry = data["models"]["oruk_fourier_2b_mlx_8bit"]
        self.assertEqual(entry["repo_id"], "oruk/oruk-fourier-2b")
        self.assertEqual(entry["subfolder"], "mlx/8bit")
        self.assertEqual(entry["runtime"], "mlx_audio")
        self.assertEqual(entry["adapter"], "qwen3")
        self.assertEqual(entry["hf_url"], "https://huggingface.co/oruk/oruk-fourier-2b")

    def test_subfolder_resolution_fetches_root_config(self):
        adapter = Qwen3Adapter(
            model_path="oruk/oruk-fourier-2b",
            precision="8-bit",
            runtime="mlx_audio",
            subfolder="mlx/8bit",
        )
        with patch("huggingface_hub.snapshot_download", return_value="/tmp/fake-snap") as mock_dl:
            resolved = adapter._resolve_model_dir()
        self.assertEqual(resolved, "/tmp/fake-snap/mlx/8bit")
        mock_dl.assert_called_once_with(
            "oruk/oruk-fourier-2b",
            allow_patterns=["config.json", "mlx/8bit/*"],
        )


if __name__ == "__main__":
    unittest.main()
