# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import warnings

import pytest

from GBOpt.artifacts import ArtifactRetentionPolicy
from GBOpt.artifacts.provenance import ArtifactProvenanceError
from GBOpt.optimization.checkpointing import (
    _artifact_archive_root,
    _configure_artifact_runtime,
    _materialize_archive_file,
    _normalize_calculation_context_config,
    _normalize_failure_diagnostic_count,
    _run_artifact_provenance,
)
from GBOpt.optimization.types import GBMinimizerTypeError, GBMinimizerValueError

_TEST_CALCULATION_CONTEXT = {"calculator": {"name": "test-evaluator"}}

# --------------------------------------------------------------------------------------
# _normalize_calculation_context_config
# --------------------------------------------------------------------------------------


def test_normalize_calculation_context_config_accepts_none_without_pruning():
    assert (
        _normalize_calculation_context_config(None, retention_policy=None) is None
    )


def test_normalize_calculation_context_config_rejects_non_mapping():
    with pytest.raises(GBMinimizerTypeError, match="must be a mapping or None"):
        _normalize_calculation_context_config(
            "not-a-mapping", retention_policy=None
        )


def test_normalize_calculation_context_config_requires_context_when_pruning():
    policy = ArtifactRetentionPolicy(prune=True)

    with pytest.raises(
        GBMinimizerValueError, match="requires a non-empty calculation_context"
    ):
        _normalize_calculation_context_config(None, retention_policy=policy)


def test_normalize_calculation_context_config_accepts_context_when_pruning():
    policy = ArtifactRetentionPolicy(prune=True)

    normalized = _normalize_calculation_context_config(
        _TEST_CALCULATION_CONTEXT, retention_policy=policy
    )

    assert normalized == _TEST_CALCULATION_CONTEXT


# --------------------------------------------------------------------------------------
# _normalize_failure_diagnostic_count
# --------------------------------------------------------------------------------------


def test_normalize_failure_diagnostic_count_accepts_nonnegative_int():
    assert _normalize_failure_diagnostic_count(5) == 5
    assert _normalize_failure_diagnostic_count(0) == 0


@pytest.mark.parametrize("value", [True, False, 1.5, "3"])
def test_normalize_failure_diagnostic_count_rejects_non_integer(value):
    with pytest.raises(GBMinimizerTypeError, match="non-Boolean integer"):
        _normalize_failure_diagnostic_count(value)


def test_normalize_failure_diagnostic_count_rejects_negative():
    with pytest.raises(GBMinimizerValueError, match="must be non-negative"):
        _normalize_failure_diagnostic_count(-1)


# --------------------------------------------------------------------------------------
# _configure_artifact_runtime
# --------------------------------------------------------------------------------------


def test_configure_artifact_runtime_defaults_to_no_store():
    cleaner, store = _configure_artifact_runtime(None, None, None)

    assert store is None
    assert cleaner is not None


def test_configure_artifact_runtime_rejects_invalid_policy_type():
    with pytest.raises(
        GBMinimizerTypeError, match="ArtifactRetentionPolicy or None"
    ):
        _configure_artifact_runtime("not-a-policy", None, None)


def test_configure_artifact_runtime_rejects_both_cleanup_owners(tmp_path):
    policy = ArtifactRetentionPolicy(prune=True)

    with pytest.raises(GBMinimizerValueError, match="not both"):
        _configure_artifact_runtime(policy, tmp_path, lambda request: None)


def test_configure_artifact_runtime_requires_prune_for_cleanup_configuration(
    tmp_path,
):
    with pytest.raises(GBMinimizerValueError, match="requires retention_policy prune"):
        _configure_artifact_runtime(None, tmp_path, None)


def test_configure_artifact_runtime_requires_cleanup_owner_when_pruning():
    policy = ArtifactRetentionPolicy(prune=True)

    with pytest.raises(
        GBMinimizerValueError,
        match="requires managed_artifact_root or cleanup_candidate",
    ):
        _configure_artifact_runtime(policy, None, None)


def test_configure_artifact_runtime_builds_store_when_policy_given(tmp_path):
    policy = ArtifactRetentionPolicy(prune=True)

    cleaner, store = _configure_artifact_runtime(policy, tmp_path, None)

    assert cleaner is not None
    assert store is not None


# --------------------------------------------------------------------------------------
# _run_artifact_provenance
# --------------------------------------------------------------------------------------


def test_run_artifact_provenance_returns_false_when_disabled():
    assert _run_artifact_provenance(None, lambda: (_ for _ in ()).throw(AssertionError)) is False


def test_run_artifact_provenance_runs_action_and_returns_true():
    calls = []

    result = _run_artifact_provenance(object(), lambda: calls.append("ran"))

    assert result is True
    assert calls == ["ran"]


def test_run_artifact_provenance_warns_and_returns_false_on_provenance_error():
    def _fail():
        raise ArtifactProvenanceError("boom")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run_artifact_provenance(object(), _fail)

    assert result is False
    assert any("Artifact provenance update failed" in str(w.message) for w in caught)


# --------------------------------------------------------------------------------------
# _artifact_archive_root
# --------------------------------------------------------------------------------------


def test_artifact_archive_root_uses_checkpoint_parent_when_available(tmp_path):
    checkpoint_file = tmp_path / "run.checkpoint.json"

    root = _artifact_archive_root(checkpoint_file, fallback_stem="unused")

    assert root == tmp_path / "run.checkpoint.artifacts"


def test_artifact_archive_root_falls_back_to_cwd_stem_when_checkpointing_disabled():
    root = _artifact_archive_root(None, fallback_stem="my_run")

    assert root.name == "my_run.artifacts"


# --------------------------------------------------------------------------------------
# _materialize_archive_file
# --------------------------------------------------------------------------------------


def test_materialize_archive_file_copies_source_to_destination(tmp_path):
    source = tmp_path / "source.data"
    source.write_text("relaxed structure")
    destination = tmp_path / "archive" / "candidate.data"

    _materialize_archive_file(source, destination)

    assert destination.read_text() == "relaxed structure"


def test_materialize_archive_file_is_a_noop_when_source_is_destination(tmp_path):
    source = tmp_path / "source.data"
    source.write_text("relaxed structure")

    _materialize_archive_file(source, source)

    assert source.read_text() == "relaxed structure"


def test_materialize_archive_file_replaces_existing_destination(tmp_path):
    source = tmp_path / "source.data"
    source.write_text("new content")
    destination = tmp_path / "archive" / "candidate.data"
    destination.parent.mkdir()
    destination.write_text("stale content")

    _materialize_archive_file(source, destination)

    assert destination.read_text() == "new content"
