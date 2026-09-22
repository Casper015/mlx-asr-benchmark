"""Tests for audited metrics, scoring formulas, and zero-cost quantization."""

import csv
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestScoringAndMetrics(unittest.TestCase):
    def setUp(self):
        self.all_results_path = REPO_ROOT / "outputs" / "ALL_RESULTS.csv"
        self.hour_results_path = REPO_ROOT / "outputs" / "HOUR_RESULTS.csv"

        with open(self.all_results_path, "r", encoding="utf-8") as f:
            self.results = list(csv.DictReader(f))

        with open(self.hour_results_path, "r", encoding="utf-8") as f:
            self.hour_results = list(csv.DictReader(f))

    def test_pipecat_count_and_tagging(self):
        """Verify 19 evaluated conditions and per-turn segmentation tag."""
        self.assertEqual(len(self.results), 19, "Must contain exactly 19 Pipecat conditions")
        for r in self.results:
            self.assertEqual(r["segmentation_method"], "None (Per-turn)")

    def test_zero_cost_quantization(self):
        """Verify that Qwen3 8-bit, 6-bit, and 5-bit have identical 0.0217 WER."""
        qwen_map = {r["condition"]: float(r["wer_normalized"]) for r in self.results}

        self.assertAlmostEqual(qwen_map["qwen3-asr-1.7b-8bit-mc"], 0.0217, places=4)
        self.assertAlmostEqual(qwen_map["qwen3-asr-1.7b-6bit-mc"], 0.0217, places=4)
        self.assertAlmostEqual(qwen_map["qwen3-asr-1.7b-5bit-mc"], 0.0217, places=4)
        # Verify BF16 does not outperform quantized variants
        self.assertGreaterEqual(qwen_map["qwen3-asr-1.7b-bf16-mc"], 0.0217)

    def test_continuous_audited_ranking(self):
        """Verify that continuous audited score highlights Parakeet at #1 and Alkd at #2."""
        # Sort valid conditions by continuous_audited
        valid = [r for r in self.results if r["condition"] != "qwen3-asr-1.7b-4bit"]
        valid.sort(key=lambda x: float(x["continuous_audited"]), reverse=True)

        rank_1 = valid[0]
        rank_2 = valid[1]

        self.assertEqual(
            rank_1["condition"],
            "parakeet-tdt-0.6b-v3",
            f"Expected Parakeet at #1, got {rank_1['condition']}"
        )
        self.assertAlmostEqual(float(rank_1["continuous_audited"]), 0.882, places=2)

        self.assertEqual(
            rank_2["condition"],
            "qwen3-asr-1.7b-4bit-alkd-a8t4",
            f"Expected Alkd mixed precision at #2, got {rank_2['condition']}"
        )
        self.assertAlmostEqual(float(rank_2["continuous_audited"]), 0.662, places=2)

    def test_defects_and_notes_flagged(self):
        """Ensure aufklarer template defect, VibeVoice prefix, and Parakeet empty tokens are flagged."""
        res_map = {r["condition"]: r for r in self.results}

        # Aufklarer 4-bit template defect
        aufk = res_map["qwen3-asr-1.7b-4bit"]
        self.assertIn("DEFECTIVE", aufk["notes"])
        self.assertIn("186/1000", aufk["notes"])

        # VibeVoice prefix notes
        vv = res_map["vibevoice-streaming-7b-4bit"]
        self.assertIn("0.0679", vv["notes"])

        # Parakeet notes
        para = res_map["parakeet-tdt-0.6b-v3"]
        self.assertIn("8/1000 empty", para["notes"])

    def test_hour_results_segmentation_methods(self):
        """Verify longform segmentation methods are explicitly tagged."""
        self.assertEqual(len(self.hour_results), 8, "Must contain 8 long-form evaluation records")
        for r in self.hour_results:
            p = r["protocol"]
            seg_method = r["segmentation_method"]
            if p == "native_continuous":
                self.assertEqual(seg_method, "None (Native Continuous)")
            elif r["run_id"] == "segvad60":
                self.assertEqual(seg_method, "Silero VAD (60s)")
            else:
                self.assertEqual(seg_method, "RMS Energy (60s)")


if __name__ == "__main__":
    unittest.main()
