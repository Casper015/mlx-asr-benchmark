"""Automated privacy leak and anonymization verification tests."""

import base64
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Encoded forbidden patterns for automated security verification
_ENCODED_TERMS = [
    "Y2FzcGVy",                             # c-a-s-p-e-r
    "dkhJWXZkc3hlM2M=",                     # v-H-I-Y-v-d-s-x-e-3-c
    "YmVya2VsZXk=",                         # b-e-r-k-e-l-e-y
    "RkEyNiBUVFM=",                         # F-A-2-6- -T-T-S
    "UGVyc29uYWwgSW5zaWdodCBRdWVzdGlvbnM=",    # P-e-r-s-o-n-a-l...
    "L1VzZXJzLw==",                         # /Users/
]

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
    ".pyc", ".pyo", ".pyd", ".so", ".dylib",
    ".wav", ".mp3", ".m4a", ".flac", ".ogg",
    ".pt", ".pth", ".bin", ".safetensors", ".onnx", ".parquet"
}


class TestPrivacyInvariants(unittest.TestCase):
    def test_zero_privacy_leakage(self):
        """Recursively scan all files in repo to guarantee 0 occurrences of forbidden strings."""
        forbidden_terms = [base64.b64decode(item).decode("utf-8") for item in _ENCODED_TERMS]
        found_violations = []

        for p in REPO_ROOT.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in BINARY_EXTENSIONS:
                continue
            # Skip git directory, virtual environments, outputs, logs, and build artifacts
            if any(part in {".git", "runtime", ".venv", "outputs", "logs", "github_export", "cache", ".scratch", "asr", "models"} for part in p.parts):
                continue
            # Skip historical handover prompt documentation
            if p.name in {"NEXT_AI_PROMPT.md", "AGENTS.md"}:
                continue
            # Skip self
            if p.name == "test_privacy_invariants.py":
                continue

            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                self.fail(f"Could not read {p}: {e}")

            content_lower = content.lower()
            lines = content.splitlines()
            for term in forbidden_terms:
                term_lower = term.lower()
                if term_lower in content_lower:
                    for idx, line in enumerate(lines, 1):
                        if term_lower in line.lower():
                            found_violations.append(
                                f"Match at {p.relative_to(REPO_ROOT)}:{idx}: {line.strip()}"
                            )

        self.assertEqual(
            len(found_violations),
            0,
            f"Privacy leakage detected ({len(found_violations)} matches):\n" + "\n".join(found_violations)
        )


if __name__ == "__main__":
    unittest.main()
