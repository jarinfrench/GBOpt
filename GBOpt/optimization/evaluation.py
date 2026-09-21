# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Candidate evaluation and checkpoint-state normalization helpers."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from GBOpt._explicit_ownership_evaluation import CandidateEvaluation
from GBOpt.FileGrainOwnership import CandidateFileMapping, GrainOwnershipError
from GBOpt.optimization.errors import GBMinimizerError, GBMinimizerValueError


@dataclass(frozen=True, slots=True)
class _CachedEvaluation:
    """Reusable result for one unchanged legacy-path carryover candidate."""

    energy: float
    structure_path: str


@dataclass(frozen=True, slots=True)
class _FailureDiagnostic:
    """Persist bounded failed-evaluation diagnostic source metadata.

    :param candidate_id: Stable logical candidate identity.
    :param generation: GA generation where the evaluation failed.
    :param input_index: Candidate position within the submitted generation.
    :param failure_reason: Durable evaluator/reconstruction failure context.
    :param source_path: Evaluator-returned diagnostic source path, when available.
    """

    candidate_id: str
    generation: int
    input_index: int
    failure_reason: str
    source_path: str | None

    @classmethod
    def from_evaluation(
        cls,
        record: CandidateEvaluation,
        *,
        generation: int,
    ) -> "_FailureDiagnostic":
        """Build one diagnostic record from a failed typed evaluation.

        :param record: Failed explicit-ownership evaluation.
        :param generation: Keyword argument, required. GA generation index.
        :return: Detached failed-evaluation diagnostic metadata.
        :raises GBMinimizerValueError: If ``record`` succeeded or generation is invalid.
        """
        if not isinstance(record, CandidateEvaluation) or record.success:
            raise GBMinimizerValueError(
                "failure diagnostics require a failed CandidateEvaluation"
            )
        if (
            isinstance(generation, (bool, np.bool_))
            or not isinstance(generation, Integral)
            or generation < 0
        ):
            raise GBMinimizerValueError(
                "failure diagnostic generation must be a non-negative integer"
            )
        return cls(
            candidate_id=record.candidate_id,
            generation=int(generation),
            input_index=record.input_index,
            failure_reason=record.failure_reason or "unknown evaluation failure",
            source_path=record.structure_path,
        )

    def to_state(self) -> dict[str, object]:
        """Return deterministic JSON-safe diagnostic state."""
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "input_index": self.input_index,
            "failure_reason": self.failure_reason,
            "source_path": self.source_path,
        }

    @classmethod
    def from_state(cls, state: object) -> "_FailureDiagnostic":
        """Restore one diagnostic record from checkpoint state.

        :param state: JSON-decoded diagnostic state.
        :return: Validated failed-evaluation diagnostic metadata.
        :raises GBMinimizerError: If state is malformed.
        """
        if not isinstance(state, dict):
            raise GBMinimizerError(
                "failure diagnostic checkpoint state must be a dictionary"
            )
        try:
            candidate_id = state["candidate_id"]
            generation = state["generation"]
            input_index = state["input_index"]
            failure_reason = state["failure_reason"]
            source_path = state.get("source_path")
        except KeyError as exc:
            raise GBMinimizerError(
                "failure diagnostic checkpoint state is incomplete"
            ) from exc
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise GBMinimizerError(
                "failure diagnostic candidate_id must be a non-empty string"
            )
        for value, name in (
            (generation, "generation"),
            (input_index, "input_index"),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise GBMinimizerError(
                    f"failure diagnostic {name} must be a non-negative integer"
                )
        if not isinstance(failure_reason, str) or not failure_reason:
            raise GBMinimizerError(
                "failure diagnostic failure_reason must be a non-empty string"
            )
        if source_path is not None and (
            not isinstance(source_path, str) or not source_path.strip()
        ):
            raise GBMinimizerError(
                "failure diagnostic source_path must be a non-empty string or None"
            )
        return cls(
            candidate_id=candidate_id,
            generation=int(generation),
            input_index=int(input_index),
            failure_reason=failure_reason,
            source_path=source_path,
        )


def _candidate_mapping_to_state(mapping: CandidateFileMapping) -> dict:
    """Serialize a candidate-local ownership mapping for checkpoint persistence.

    :param mapping: Validated candidate/file mapping.
    :return: JSON-safe mapping state without live optimizer objects.
    """
    return {
        "atom_ids": mapping.atom_ids,
        "labels": mapping.labels,
        "species": mapping.species.tolist(),
        "box_dims": mapping.box_dims,
        "gb_plane_x": mapping.gb_plane_x,
        "inplane_periodic": mapping.inplane_periodic,
        "left_grain_x_bounds": mapping.left_grain_x_bounds,
        "right_grain_x_bounds": mapping.right_grain_x_bounds,
        "coordinate_tolerance": mapping.coordinate_tolerance,
        "normal_topology": mapping.normal_topology.value,
    }


def _candidate_mapping_from_state(state: object) -> CandidateFileMapping:
    """Reconstruct and validate a checkpointed candidate/file mapping.

    :param state: Deserialized mapping state.
    :return: Validated candidate-local ownership mapping.
    :raises GrainOwnershipError: If the checkpointed mapping is malformed.
    """
    if not isinstance(state, dict):
        raise GrainOwnershipError("candidate mapping state must be a dictionary")
    try:
        return CandidateFileMapping(
            atom_ids=np.asarray(state["atom_ids"], dtype=object),
            labels=np.asarray(state["labels"], dtype=object),
            species=np.asarray(state["species"], dtype=object),
            box_dims=np.asarray(state["box_dims"], dtype=object),
            gb_plane_x=state["gb_plane_x"],
            inplane_periodic=tuple(state["inplane_periodic"]),
            left_grain_x_bounds=np.asarray(
                state["left_grain_x_bounds"], dtype=object
            ),
            right_grain_x_bounds=np.asarray(
                state["right_grain_x_bounds"], dtype=object
            ),
            coordinate_tolerance=state["coordinate_tolerance"],
            normal_topology=state["normal_topology"],
        )
    except (KeyError, TypeError) as exc:
        raise GrainOwnershipError(
            "candidate mapping checkpoint state is incomplete or malformed"
        ) from exc
