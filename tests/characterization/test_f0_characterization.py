"""Regression tests for the committed F0 behavior baseline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from characterization.f0_manifest import (
    SCHEMA_VERSION,
    SOURCE_ARCHIVE,
    SOURCE_ARCHIVE_SHA256,
    behavior_manifest,
)


class TestF0CharacterizationBaseline(unittest.TestCase):
    """Freeze cross-cutting behavior before production refactoring begins."""

    @classmethod
    def setUpClass(cls) -> None:
        manifest_path = Path(__file__).with_name("baseline_manifest.json")
        cls.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_baseline_identity_is_recorded(self) -> None:
        self.assertEqual(self.manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            self.manifest["baseline"]["source_archive"], SOURCE_ARCHIVE
        )
        self.assertEqual(
            self.manifest["baseline"]["source_archive_sha256"],
            SOURCE_ARCHIVE_SHA256,
        )

    def test_current_behavior_matches_committed_manifest(self) -> None:
        self.assertEqual(behavior_manifest(), self.manifest["behavior"])


if __name__ == "__main__":
    unittest.main()
