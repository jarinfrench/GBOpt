# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define the algorithm-neutral evaluation result contract and its value types.

This module consumes a candidate's physical and optimizer-selection energies, its
success/failure status, and an optional reconstructed structure artifact. It returns one
canonical immutable ``EvaluationResult`` per candidate evaluation. Callback invocation,
artifact reconstruction, and legacy-shape adaptation do not belong here; optimizer
selection, acceptance, and population policy do not either.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from GBOpt.GBManipulator import GBManipulator


class EvaluationError(Exception):
    """Base class for evaluation-subsystem errors."""


class EvaluationTypeError(EvaluationError, TypeError):
    """Raised when evaluation-domain state has an invalid type."""


class EvaluationValueError(EvaluationError, ValueError):
    """Raised when evaluation-domain state has an invalid value."""


def _normalize_candidate_id(candidate_id: object) -> str:
    """Validate one stable candidate identity.

    :param candidate_id: Candidate identity to validate.
    :return: Validated non-empty identity.
    :raises EvaluationTypeError: If ``candidate_id`` is not a non-empty string.
    """
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise EvaluationTypeError("candidate_id must be a non-empty string")
    return candidate_id


def _normalize_input_index(input_index: object) -> int:
    """Normalize one candidate population index.

    :param input_index: Candidate position to normalize.
    :return: Python integer candidate index.
    :raises EvaluationTypeError: If ``input_index`` is Boolean or non-integral.
    """
    if isinstance(input_index, (bool, np.bool_)) or not isinstance(
        input_index, Integral
    ):
        raise EvaluationTypeError("input_index must be a non-Boolean integer")
    return int(input_index)


def _normalize_energy(value: object, *, name: str) -> float:
    """Normalize one finite energy-like scalar.

    :param value: Energy-like value to normalize.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Finite Python float.
    :raises EvaluationTypeError: If ``value`` is Boolean or non-real.
    :raises EvaluationValueError: If ``value`` is non-finite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise EvaluationTypeError(f"{name} must be a non-Boolean real scalar")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise EvaluationValueError(f"{name} must be finite")
    return normalized


class EvaluationStatus(str, Enum):
    """Stable outcome of one candidate evaluation, independent of algorithm."""

    SUCCESS = "success"
    FAILED = "failed"


class FailureStage(str, Enum):
    """Pipeline stage a failed evaluation's diagnostic context originates from."""

    EVALUATOR = "evaluator"
    ARTIFACT = "artifact"
    PARSE = "parse"
    OWNERSHIP = "ownership"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class StructureArtifact:
    """Reference to one on-disk structure file, without embedding its contents.

    :param path: Canonical structure file path.
    :param format: Structure file format identifier (e.g. ``"lammps"``).
    :param digest: Stable content identifier, when computed; ``None`` otherwise.
    :raises EvaluationTypeError: If path, format, or digest has an invalid type.
    :raises EvaluationValueError: If path or format is empty.
    """

    path: str
    format: str
    digest: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate artifact reference fields.

        :raises EvaluationTypeError: If path, format, or digest has an invalid type.
        :raises EvaluationValueError: If path or format is empty.
        """
        if not isinstance(self.path, (str, Path)):
            raise EvaluationTypeError("path must be a path-like string")
        path = str(self.path)
        if not path.strip():
            raise EvaluationValueError("path must not be empty")
        if not isinstance(self.format, str):
            raise EvaluationTypeError("format must be a string")
        if not self.format.strip():
            raise EvaluationValueError("format must not be empty")
        if self.digest is not None and (
            not isinstance(self.digest, str) or not self.digest.strip()
        ):
            raise EvaluationTypeError("digest must be a non-empty string or None")
        object.__setattr__(self, "path", path)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        format: str,
        *,
        compute_digest: bool = False,
    ) -> StructureArtifact:
        """Build one artifact reference, optionally hashing its current contents.

        :param path: Structure file path.
        :param format: Structure file format identifier.
        :param compute_digest: Keyword argument, optional, defaults to ``False``. Hash
            the file's current contents into a stable digest.
        :return: Validated artifact reference.
        :raises EvaluationTypeError: If path, format, or digest has an invalid type.
        :raises EvaluationValueError: If path or format is empty.
        :raises OSError: If ``compute_digest`` is set and the file cannot be read.
        """
        digest = None
        if compute_digest:
            import hashlib

            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return cls(path=str(path), format=format, digest=digest)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Canonical, algorithm-neutral result for one candidate evaluation.

    ``selection_energy`` is the value an optimizer uses for acceptance/selection
    (the physical energy on success, or an optimizer-supplied penalty on failure).
    ``energy`` is the physical energy alone, populated only on success -- the two are
    kept as distinct concepts even though they share a value in the successful case.

    :param candidate_id: Stable logical candidate identity independent of artifact
        paths.
    :param input_index: Candidate position in the submitted population.
    :param status: Stable success/failure outcome.
    :param selection_energy: Optimizer-facing energy used for acceptance/selection.
    :param energy: Physical energy, populated only when ``status`` is ``SUCCESS``.
    :param artifact: Reconstructed structure artifact reference, when available.
    :param manipulator: Validated reconstructed candidate, when successful.
    :param failure_stage: Pipeline stage the failure originated from, required when
        ``status`` is ``FAILED``.
    :param failure_code: Stable machine-readable failure identifier, when available.
    :param failure_message: Human-readable failure context, required when ``status`` is
        ``FAILED``.
    :raises EvaluationTypeError: If scalar, status, or reference fields have invalid
        types.
    :raises EvaluationValueError: If energies are non-finite or success/failure fields
        are internally inconsistent.
    """

    candidate_id: str
    input_index: int
    status: EvaluationStatus
    selection_energy: float
    energy: float | None = None
    artifact: StructureArtifact | None = None
    manipulator: GBManipulator | None = None
    failure_stage: FailureStage | None = None
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        """Normalize scalar fields and enforce coherent success/failure state.

        :raises EvaluationTypeError: If scalar, status, or reference fields have
            invalid types.
        :raises EvaluationValueError: If energies are non-finite or success/failure
            fields are internally inconsistent.
        """
        candidate_id = _normalize_candidate_id(self.candidate_id)
        input_index = _normalize_input_index(self.input_index)
        if not isinstance(self.status, EvaluationStatus):
            raise EvaluationTypeError("status must be an EvaluationStatus")
        selection_energy = _normalize_energy(
            self.selection_energy, name="selection_energy"
        )
        if self.artifact is not None and not isinstance(
            self.artifact, StructureArtifact
        ):
            raise EvaluationTypeError("artifact must be a StructureArtifact or None")

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "input_index", input_index)
        object.__setattr__(self, "selection_energy", selection_energy)

        if self.status is EvaluationStatus.SUCCESS:
            if self.energy is None:
                raise EvaluationValueError(
                    "successful evaluation requires a physical energy"
                )
            energy = _normalize_energy(self.energy, name="energy")
            object.__setattr__(self, "energy", energy)
            if self.artifact is None:
                raise EvaluationValueError(
                    "successful evaluation requires a structure artifact"
                )
            if (
                self.failure_stage is not None
                or self.failure_code is not None
                or self.failure_message is not None
            ):
                raise EvaluationValueError(
                    "successful evaluation must not include failure context"
                )
            return

        if self.energy is not None:
            raise EvaluationValueError("failed evaluation must not include an energy")
        if self.manipulator is not None:
            raise EvaluationValueError(
                "failed evaluation must not include a manipulator"
            )
        if not isinstance(self.failure_stage, FailureStage):
            raise EvaluationValueError("failed evaluation requires a failure_stage")
        if not isinstance(self.failure_message, str) or not self.failure_message:
            raise EvaluationValueError("failed evaluation requires a failure_message")
        if self.failure_code is not None and (
            not isinstance(self.failure_code, str) or not self.failure_code
        ):
            raise EvaluationTypeError(
                "failure_code must be a non-empty string or None"
            )
