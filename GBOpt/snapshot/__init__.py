# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the typed schema-v2 restart snapshot contract and its schema-v1 migrator.

The package-level surface contains the immutable snapshot value types describing all
restart-critical Monte Carlo/genetic-algorithm state, plus the strict schema-v1 reader
that migrates an existing schema-v1 checkpoint (JSON or pickle) into a validated
snapshot. No MC/GA loop is wired through these snapshots or the migrator here -- they
exist as a standalone, independently testable seam for a later step to adopt.
"""

from .migration import (
    SnapshotMigrationError,
    migrate_genetic_algorithm_checkpoint,
    migrate_monte_carlo_checkpoint,
)
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
    "SnapshotMigrationError",
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
    # Migrator
    "migrate_monte_carlo_checkpoint",
    "migrate_genetic_algorithm_checkpoint",
]
