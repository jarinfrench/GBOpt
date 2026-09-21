# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared artifact-retention and checkpoint-runtime helpers for MC and GA."""

import os
import shutil
import warnings
from collections.abc import Callable, Mapping
from numbers import Integral
from pathlib import Path

import numpy as np

from GBOpt.artifacts.cleanup import (
    ArtifactCleanupError,
    ArtifactCleanupRequest,
    _ArtifactCleaner,
    remove_managed_path,
)
from GBOpt.artifacts.policy import ArtifactRetentionPolicy
from GBOpt.artifacts.provenance import (
    ArtifactProvenanceError,
    _ArtifactProvenance,
    _normalize_calculation_context,
)
from GBOpt.artifacts.store import ArtifactStore, ArtifactStoreError
from GBOpt.artifacts.types import ArtifactPin, CandidatePropertyContext
from GBOpt.optimization.types import (
    GBMinimizerError,
    GBMinimizerTypeError,
    GBMinimizerValueError,
)


def _normalize_calculation_context_config(
    calculation_context: object,
    *,
    retention_policy: ArtifactRetentionPolicy | None,
) -> dict[str, object] | None:
    """Validate run-level calculation provenance at the minimizer boundary.

    :param calculation_context: Evaluator/campaign provenance mapping or ``None``.
    :param retention_policy: Keyword argument, required. Configured retention policy.
    :return: Detached normalized calculation context, or ``None``.
    :raises GBMinimizerTypeError: If ``calculation_context`` is not a mapping or
        ``None``.
    :raises GBMinimizerValueError: If provenance values are invalid or pruning lacks a
        non-empty calculation context.
    """
    if calculation_context is not None and not isinstance(
        calculation_context, Mapping
    ):
        raise GBMinimizerTypeError(
            "calculation_context must be a mapping or None"
        )
    try:
        normalized = _normalize_calculation_context(calculation_context)
    except ArtifactProvenanceError as exc:
        raise GBMinimizerValueError(str(exc)) from exc
    if (
        retention_policy is not None
        and retention_policy.prune
        and not normalized
    ):
        raise GBMinimizerValueError(
            "retention_policy prune=True requires a non-empty calculation_context"
        )
    return normalized


def _normalize_failure_diagnostic_count(value: object) -> int:
    """Validate the bounded failed-evaluation diagnostic count.

    :param value: Maximum number of recent failed evaluator sources to retain.
    :return: Non-negative Python integer bound.
    :raises GBMinimizerTypeError: If ``value`` is Boolean or non-integral.
    :raises GBMinimizerValueError: If ``value`` is negative.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise GBMinimizerTypeError(
            "failure_diagnostic_count must be a non-Boolean integer"
        )
    normalized = int(value)
    if normalized < 0:
        raise GBMinimizerValueError(
            "failure_diagnostic_count must be non-negative"
        )
    return normalized


def _configure_artifact_runtime(
    retention_policy: ArtifactRetentionPolicy | None,
    managed_artifact_root: str | Path | None,
    cleanup_candidate: Callable[[ArtifactCleanupRequest], None] | None,
) -> tuple[_ArtifactCleaner, ArtifactStore | None]:
    """Validate shared artifact configuration and construct runtime helpers.

    :param retention_policy: Scientific retention policy, or ``None`` for legacy
        keep-all behavior.
    :param managed_artifact_root: Optional evaluator path root owned by GBOpt.
    :param cleanup_candidate: Optional evaluator-owned cleanup callback.
    :return: Configured cleanup dispatcher and optional artifact store.
    :raises GBMinimizerTypeError: If policy, path, or cleanup callback types are
        invalid.
    :raises GBMinimizerValueError: If cleanup ownership is ambiguous or inconsistent
        with pruning configuration.
    """
    if retention_policy is not None and not isinstance(
        retention_policy, ArtifactRetentionPolicy
    ):
        raise GBMinimizerTypeError(
            "retention_policy must be an ArtifactRetentionPolicy or None"
        )
    if managed_artifact_root is not None and not isinstance(
        managed_artifact_root, (str, os.PathLike)
    ):
        raise GBMinimizerTypeError(
            "managed_artifact_root must be a path-like value or None"
        )
    if cleanup_candidate is not None and not callable(cleanup_candidate):
        raise GBMinimizerTypeError("cleanup_candidate must be callable or None")
    cleanup_configured = (
        managed_artifact_root is not None or cleanup_candidate is not None
    )
    if managed_artifact_root is not None and cleanup_candidate is not None:
        raise GBMinimizerValueError(
            "configure either managed_artifact_root or cleanup_candidate, not both"
        )
    if cleanup_configured and (
        retention_policy is None or not retention_policy.prune
    ):
        raise GBMinimizerValueError(
            "artifact cleanup configuration requires retention_policy prune=True"
        )
    if (
        retention_policy is not None
        and retention_policy.prune
        and not cleanup_configured
    ):
        raise GBMinimizerValueError(
            "retention_policy prune=True requires managed_artifact_root or "
            "cleanup_candidate"
        )
    try:
        cleaner = _ArtifactCleaner(
            managed_artifact_root=managed_artifact_root,
            cleanup_candidate=cleanup_candidate,
        )
        store = (
            ArtifactStore(policy=retention_policy)
            if retention_policy is not None
            else None
        )
    except (ArtifactCleanupError, ArtifactStoreError) as exc:
        raise GBMinimizerValueError(str(exc)) from exc
    return cleaner, store


def _run_artifact_provenance(
    provenance: _ArtifactProvenance | None,
    action: Callable[[], None],
) -> bool:
    """Run one non-authoritative provenance write with warning-only failure policy.

    :param provenance: Active provenance writer, or ``None`` when disabled.
    :param action: Zero-argument provenance operation to execute.
    :return: Whether the provenance operation completed successfully.
    """
    if provenance is None:
        return False
    try:
        action()
    except ArtifactProvenanceError as exc:
        warnings.warn(
            f"Artifact provenance update failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def _register_retention_candidate(
    *,
    artifact_store: ArtifactStore,
    retention_policy: ArtifactRetentionPolicy,
    context: CandidatePropertyContext,
    source_path: str | Path,
    lineage: tuple[str, ...],
    provenance: _ArtifactProvenance | None,
) -> None:
    """Register one validated relaxed candidate and record retention deltas.

    :param artifact_store: Keyword argument, required. Runtime artifact reference store.
    :param retention_policy: Keyword argument, required. Scientific policy used to
        acquire candidate properties.
    :param context: Keyword argument, required. Validated relaxed physical candidate
        state.
    :param source_path: Keyword argument, required. Evaluator-returned candidate
        artifact path.
    :param lineage: Keyword argument, required. Stable logical parent candidate
        identities.
    :param provenance: Keyword argument, required. Optional non-authoritative provenance
        writer.
    :raises ArtifactPolicyError: If property acquisition or rule evaluation fails.
    :raises ArtifactStoreError: If artifact-store state is invalid or conflicting.
    :raises ArtifactValueError: If candidate property state is malformed.
    """
    if context.candidate_id in artifact_store:
        return
    before_reasons = {
        artifact.candidate_id: set(artifact.retention_reasons)
        for artifact in artifact_store.records()
    }
    candidate = retention_policy.candidate_from_context(context, lineage=lineage)
    artifact_store.register_candidate(candidate, source_path=source_path)
    after_reasons = {
        artifact.candidate_id: set(artifact.retention_reasons)
        for artifact in artifact_store.records()
    }

    _run_artifact_provenance(
        provenance, lambda: provenance.record_candidate_evaluated(candidate)
    )
    _run_artifact_provenance(
        provenance, lambda: provenance.record_properties_calculated(candidate)
    )
    for candidate_id in sorted(set(before_reasons).union(after_reasons)):
        previous = before_reasons.get(candidate_id, set())
        current = after_reasons.get(candidate_id, set())
        for reason in sorted(current.difference(previous)):
            _run_artifact_provenance(
                provenance,
                lambda candidate_id=candidate_id, reason=reason: (
                    provenance.record_retention_reason_added(candidate_id, reason)
                ),
            )
        for reason in sorted(previous.difference(current)):
            _run_artifact_provenance(
                provenance,
                lambda candidate_id=candidate_id, reason=reason: (
                    provenance.record_retention_reason_removed(candidate_id, reason)
                ),
            )


def _artifact_archive_root(checkpoint_file: Path | None, *, fallback_stem: str) -> Path:
    """Return the run-owned artifact archive root.

    :param checkpoint_file: Run checkpoint path, or ``None`` when disabled.
    :param fallback_stem: Keyword argument, required. Archive directory stem used when
        checkpointing is disabled.
    :return: Deterministic run-owned artifact root.
    """
    if checkpoint_file is not None:
        return checkpoint_file.parent / f"{checkpoint_file.stem}.artifacts"
    return Path.cwd() / f"{fallback_stem}.artifacts"


def _materialize_archive_file(source: Path, destination: Path) -> None:
    """Atomically hard-link or copy one canonical retained structure.

    :param source: Existing evaluator-returned structure file.
    :param destination: Canonical archive destination.
    :raises OSError: If directory creation, linking, copying, or replacement fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def _cleanup_prunable_sources(
    artifact_store: ArtifactStore,
    cleaner: _ArtifactCleaner,
    provenance: _ArtifactProvenance | None,
) -> None:
    """Best-effort cleanup of committed evaluator sources reported as prunable.

    Cleanup failures leak storage and emit diagnostics; they never invalidate committed
    optimizer state.

    :param artifact_store: Runtime artifact reference store.
    :param cleaner: Explicit evaluator-source cleanup dispatcher.
    :param provenance: Optional non-authoritative provenance writer.
    """
    try:
        records = artifact_store.records()
    except ArtifactStoreError as exc:
        warnings.warn(
            f"Artifact cleanup state could not be inspected: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    for artifact in records:
        try:
            if not artifact_store.source_is_prunable(artifact.candidate_id):
                continue
            if artifact.source_path is None:
                continue
            request = ArtifactCleanupRequest(
                candidate_id=artifact.candidate_id,
                source_path=Path(artifact.source_path),
                archive_path=(
                    None
                    if artifact.archive_path is None
                    else Path(artifact.archive_path)
                ),
            )
            cleaner.cleanup_source(request)
            _run_artifact_provenance(
                provenance,
                lambda artifact=artifact: provenance.record_source_pruned(
                    artifact.candidate_id, artifact.source_path
                ),
            )
        except (ArtifactCleanupError, ArtifactStoreError) as exc:
            if artifact.source_path is not None:
                _run_artifact_provenance(
                    provenance,
                    lambda artifact=artifact, exc=exc: provenance.record_cleanup_failed(
                        "source_prune",
                        artifact.source_path,
                        str(exc),
                        candidate_id=artifact.candidate_id,
                    ),
                )
            warnings.warn(
                f"Artifact cleanup failed for candidate {artifact.candidate_id!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _remove_archive_evictions(
    archive_evictions: list[tuple[str, str]],
    *,
    archive_root: Path,
    provenance: _ArtifactProvenance | None,
) -> None:
    """Best-effort removal of canonical archives detached before checkpoint commit.

    :param archive_evictions: Candidate IDs and canonical paths detached from store
        state.
    :param archive_root: Keyword argument, required. Run-owned containment root.
    :param provenance: Keyword argument, required. Optional provenance writer.
    """
    seen: set[Path] = set()
    for candidate_id, raw_path in archive_evictions:
        path = Path(raw_path)
        if path in seen:
            continue
        seen.add(path)
        try:
            remove_managed_path(path, managed_root=archive_root)
            _run_artifact_provenance(
                provenance,
                lambda candidate_id=candidate_id, path=path: (
                    provenance.record_archive_evicted(candidate_id, path)
                ),
            )
        except ArtifactCleanupError as exc:
            _run_artifact_provenance(
                provenance,
                lambda candidate_id=candidate_id, path=path, exc=exc: (
                    provenance.record_cleanup_failed(
                        "archive_evict",
                        path,
                        str(exc),
                        candidate_id=candidate_id,
                    )
                ),
            )
            warnings.warn(
                f"Artifact cleanup failed for archived structure {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _prepare_archive_state(
    artifact_store: ArtifactStore | None,
    materialize_archive: Callable[[str], None],
    *,
    archive_detached: Callable[[str], None] | None = None,
) -> list[tuple[str, str]]:
    """Materialize required archives and detach unreferenced archive paths.

    Store state is updated before checkpoint serialization. Detached archive files are
    returned to the caller for deletion only after the checkpoint commits successfully.

    :param artifact_store: Runtime artifact store, or ``None`` when tracking is
        disabled.
    :param materialize_archive: Callback that materializes the required archive for one
        candidate and updates ``artifact_store`` with its canonical path.
    :param archive_detached: Keyword argument, optional, defaults to ``None``. Callback
        invoked after an unreferenced archive is detached from store state.
    :return: Candidate IDs and archive paths eligible for post-commit deletion.
    :raises ArtifactStoreError: If artifact-store state is invalid.
    :raises GBMinimizerError: If an already-materialized retained archive is missing.
    """
    if artifact_store is None:
        return []
    records = artifact_store.records()
    required_ids = [
        artifact.candidate_id
        for artifact in records
        if artifact.retention_reasons
        or ArtifactPin.BEST_RESULT in artifact.pins
    ]
    for candidate_id in required_ids:
        archive_path = artifact_store.archive_path(candidate_id)
        if archive_path is None:
            materialize_archive(candidate_id)
        elif not Path(archive_path).is_file():
            raise GBMinimizerError(
                f"retained archive path {archive_path} is missing"
            )

    evictions: list[tuple[str, str]] = []
    for artifact in artifact_store.records():
        if artifact.archive_path is None:
            continue
        if artifact.retention_reasons or artifact.pins:
            continue
        evictions.append((artifact.candidate_id, artifact.archive_path))
        artifact_store.set_archive_path(artifact.candidate_id, None)
        if archive_detached is not None:
            archive_detached(artifact.candidate_id)
    return evictions


def _cleanup_committed_artifacts(
    artifact_store: ArtifactStore | None,
    cleaner: _ArtifactCleaner,
    provenance: _ArtifactProvenance | None,
    archive_evictions: list[tuple[str, str]],
    *,
    archive_root: Path,
) -> None:
    """Best-effort evaluator/archive cleanup after a durable checkpoint commit.

    :param artifact_store: Runtime artifact store, or ``None`` when tracking is
        disabled.
    :param cleaner: Explicit evaluator-source cleanup dispatcher.
    :param provenance: Optional non-authoritative provenance writer.
    :param archive_evictions: Candidate IDs and detached canonical archive paths.
    :param archive_root: Keyword argument, required. Run-owned containment root.
    """
    if artifact_store is None:
        return
    _cleanup_prunable_sources(artifact_store, cleaner, provenance)
    _remove_archive_evictions(
        archive_evictions,
        archive_root=archive_root,
        provenance=provenance,
    )


def _write_artifact_manifest(
    artifact_store: ArtifactStore | None,
    provenance: _ArtifactProvenance | None,
    *,
    ownership_metadata: dict[str, dict] | None = None,
    failure_diagnostics: tuple[dict[str, object], ...] = (),
) -> bool:
    """Best-effort persistence of current artifact state for observability.

    A successful write is also the destructive-cleanup gate: callers may remove
    evaluator/archive artifacts only after the current manifest, including required
    run-level calculation provenance, has been persisted.

    :param artifact_store: Runtime store, or ``None`` when artifact tracking is
        disabled.
    :param provenance: Provenance writer, or ``None`` when output is disabled.
    :param ownership_metadata: Keyword argument, optional, defaults to ``None``.
        Candidate reconstruction metadata for ownership-aware archives.
    :param failure_diagnostics: Keyword argument, optional. Current bounded failed
        evaluator-source diagnostics retained outside ``ArtifactStore``.
    :return: Whether required current-state provenance was persisted successfully.
    """
    if artifact_store is None:
        return True
    if provenance is None:
        return False
    try:
        records = artifact_store.records()
    except ArtifactStoreError as exc:
        warnings.warn(
            f"Artifact provenance state could not be inspected: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return _run_artifact_provenance(
        provenance,
        lambda: provenance.write_manifest(
            records,
            ownership_metadata=ownership_metadata,
            failure_diagnostics=failure_diagnostics,
        ),
    )

