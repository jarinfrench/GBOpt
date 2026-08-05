"""Reference tests for checkpoint behavior absent from the current baseline.

These skipped tests deliberately name the legacy capabilities that CP0 must audit and
that CP1-CP5 may later implement.  They are not promises of raw legacy schema or pickle
compatibility.
"""

import unittest


_SKIP_REASON = (
    "Checkpoint/restart is absent from the F0 implementation baseline; this test "
    "records legacy behavior for CP0 and later checkpoint PRs."
)


class TestLegacyCheckpointBehaviorReference(unittest.TestCase):
    @unittest.skip(_SKIP_REASON)
    def test_disabled_checkpointing_creates_no_file(self) -> None:
        self.fail("Reference-only test")

    @unittest.skip(_SKIP_REASON)
    def test_interval_and_final_checkpoint_saves_are_distinct(self) -> None:
        self.fail("Reference-only test")

    @unittest.skip(_SKIP_REASON)
    def test_mc_resume_restores_rng_run_identity_and_algorithm_state(self) -> None:
        self.fail("Reference-only test")

    @unittest.skip(_SKIP_REASON)
    def test_mc_completed_run_can_be_extended(self) -> None:
        self.fail("Reference-only test")

    @unittest.skip(_SKIP_REASON)
    def test_ga_generation_resume_matches_continuous_run(self) -> None:
        self.fail("Reference-only test")

    @unittest.skip(_SKIP_REASON)
    def test_ga_intra_generation_cache_skips_completed_candidates(self) -> None:
        self.fail("Reference-only test")

    @unittest.skip(_SKIP_REASON)
    def test_ga_pending_artifacts_exist_before_snapshot_publication(self) -> None:
        self.fail("Reference-only test")

    @unittest.skip(_SKIP_REASON)
    def test_resume_fails_loudly_when_required_artifact_is_missing(self) -> None:
        self.fail("Reference-only test")


if __name__ == "__main__":
    unittest.main()
