# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the algorithm-neutral evaluation result contract and its adapters.

The package-level surface contains the canonical ``EvaluationResult`` value type, its
``EvaluationStatus``/``FailureStage`` enums and ``StructureArtifact`` reference type,
the full exception hierarchy, and adapters from the existing ownership-aware
``CandidateEvaluation`` object, legacy scalar tuple callbacks, and legacy batch
dictionaries into ``EvaluationResult``. No MC/GA loop is wired through these adapters
here -- they exist as a standalone, independently testable seam for a later step to
adopt.
"""

from .adapters import from_batch_dict, from_candidate_evaluation, from_scalar_tuple
from .types import (
    EvaluationError,
    EvaluationResult,
    EvaluationStatus,
    EvaluationTypeError,
    EvaluationValueError,
    FailureStage,
    StructureArtifact,
)

__all__ = [
    # Exceptions
    "EvaluationError",
    "EvaluationTypeError",
    "EvaluationValueError",
    # Enums
    "EvaluationStatus",
    "FailureStage",
    # Value types
    "StructureArtifact",
    "EvaluationResult",
    # Adapters
    "from_candidate_evaluation",
    "from_scalar_tuple",
    "from_batch_dict",
]
