# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the typed schema-v2 restart snapshot contract.

The package-level surface contains the immutable snapshot value types describing all
restart-critical Monte Carlo/genetic-algorithm state. No MC/GA loop is wired through
these snapshots here -- they exist as a standalone, independently testable seam for a
later step to adopt.
"""

from .types import (
    SNAPSHOT_SCHEMA_VERSION,
    CandidateEvaluationSnapshot,
    FailureDiagnosticSnapshot,
    GenerationHistoryEntrySnapshot,
    GeneticAlgorithmSnapshot,
    LineageStepSnapshot,
    MonteCarloSnapshot,
    MonteCarloStepRecordSnapshot,
    PopulationCandidateSnapshot,
    RngStateSnapshot,
    RunIdentitySnapshot,
    SnapshotError,
    SnapshotTypeError,
    SnapshotValueError,
)

__all__ = [
    # Exceptions
    "SnapshotError",
    "SnapshotTypeError",
    "SnapshotValueError",
    # Schema version
    "SNAPSHOT_SCHEMA_VERSION",
    # Value types
    "RngStateSnapshot",
    "RunIdentitySnapshot",
    "LineageStepSnapshot",
    "GenerationHistoryEntrySnapshot",
    "FailureDiagnosticSnapshot",
    "MonteCarloStepRecordSnapshot",
    "CandidateEvaluationSnapshot",
    "PopulationCandidateSnapshot",
    "MonteCarloSnapshot",
    "GeneticAlgorithmSnapshot",
]
