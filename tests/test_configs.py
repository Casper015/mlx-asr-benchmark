"""Unit tests for benchmark and model configurations."""

import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestConfigurations(unittest.TestCase):
    def setUp(self):
        self.models_path = REPO_ROOT / "configs" / "models.json"
        self.benchmark_path = REPO_ROOT / "configs" / "benchmark.json"
        self.audio_manifest_path = REPO_ROOT / "manifests" / "audio_manifest.json"

    def test_models_json_structure(self):
        """Ensure models.json exists and all models have valid Hugging Face URLs."""
        self.assertTrue(self.models_path.exists(), "models.json must exist")
        with open(self.models_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("models", data)
        models = data["models"]
        self.assertGreaterEqual(len(models), 19, "Must register at least 19 models")

        for key, entry in models.items():
            self.assertIn("name", entry, f"Missing 'name' in model {key}")
            self.assertIn("repo_id", entry, f"Missing 'repo_id' in model {key}")
            self.assertIn("hf_url", entry, f"Missing 'hf_url' in model {key}")
            self.assertTrue(
                entry["hf_url"].startswith("https://huggingface.co/"),
                f"Invalid HF URL in {key}: {entry['hf_url']}"
            )
            # Verify no hardcoded personal home paths
            local_path = entry.get("local_path", "")
            if local_path:
                self.assertFalse(local_path.startswith("/" + "Users" + "/"), f"Absolute user path found in {key}: {local_path}")

    def test_benchmark_json_specification(self):
        """Ensure benchmark parameters enforce Silero VAD default and 20GB memory ceiling."""
        self.assertTrue(self.benchmark_path.exists(), "benchmark.json must exist")
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Hardware & limits
        env = cfg.get("environment", {})
        self.assertEqual(env.get("memory_ceiling_mb"), 20480.0)

        # Default segmentation
        seg = cfg.get("segmentation", {})
        self.assertEqual(seg.get("default_method"), "silero_vad")
        self.assertIn("vad60s", seg.get("default_manifest", ""))

        # Longform acoustic specification
        longform = cfg.get("longform", {})
        desc = longform.get("description", "")
        self.assertIn("Continuous long-form audio benchmark", desc)
        self.assertIn("3,731.8s", desc)
        self.assertIn("16 kHz mono PCM WAV", desc)

    def test_audio_manifest_provenance(self):
        """Ensure audio manifest documents Pipecat provenance and long-form physical properties."""
        self.assertTrue(self.audio_manifest_path.exists(), "audio_manifest.json must exist")
        with open(self.audio_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        datasets = manifest.get("audio_datasets", {})
        self.assertIn("longform_continuous_benchmark", datasets)
        self.assertIn("pipecat_conversational_1000", datasets)

        pipecat = datasets["pipecat_conversational_1000"]
        self.assertEqual(pipecat.get("upstream_repo"), "pipecat-ai/smart-turn-data-v3.1-train")
        self.assertEqual(pipecat.get("split_file"), "train-00000-of-00001.parquet")
        self.assertEqual(pipecat.get("turn_count"), 1000)


if __name__ == "__main__":
    unittest.main()
