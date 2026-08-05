"""Adapt file-backed evaluator callbacks to explicit grain ownership.

The adapter consumes optimizer-supplied callbacks, candidate structures, ownership
mappings, and failure energy policy. It returns aligned :class:`CandidateEvaluation`
records after result normalization, artifact validation, and candidate reconstruction.
Optimizer selection and breeding policy do not belong here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from uuid import UUID

import numpy as np

from GBOpt.FileGrainOwnership import (
    CandidateFileMapping,
    GrainOwnershipError,
    LammpsDataError,
    reload_explicit_manipulator,
)
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import GBManipulator, GBManipulatorError, ParentError

_MISSING = object()


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Aligned result for one explicit-ownership candidate evaluation.

    :param input_index: Candidate position in the submitted population, or ``-1`` for
        the initial seed evaluation.
    :param energy: Normalized finite energy, including the optimizer-supplied failure
        energy when ``success`` is false.
    :param structure_path: Canonical evaluator artifact path, when available.
    :param mapping: Candidate-to-file ownership mapping, when available.
    :param manipulator: Evaluator-owned reconstructed candidate, when successful.
    :param success: Whether evaluation and reconstruction both succeeded.
    :param failure_reason: Failure context when ``success`` is false.
    :raises TypeError: If scalar fields have invalid types.
    :raises ValueError: If energy is non-finite or result fields contradict ``success``.
    """

    input_index: int
    energy: float
    structure_path: str | None
    mapping: CandidateFileMapping | None
    manipulator: GBManipulator | None
    success: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """Normalize scalar fields and enforce success/failure state coherence.

        :raises TypeError: If scalar fields have invalid types.
        :raises ValueError: If energy is non-finite or result fields contradict
            ``success``.
        """
        if isinstance(self.input_index, (bool, np.bool_)) or not isinstance(
            self.input_index, Integral
        ):
            raise TypeError("input_index must be a non-Boolean integer")
        if isinstance(self.energy, (bool, np.bool_)) or not isinstance(
            self.energy, Real
        ):
            raise TypeError("energy must be a non-Boolean real scalar")
        normalized_energy = float(self.energy)
        if not np.isfinite(normalized_energy):
            raise ValueError("energy must be finite")
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if self.structure_path is not None and not isinstance(self.structure_path, str):
            raise TypeError("structure_path must be a string or None")
        if self.failure_reason is not None and not isinstance(self.failure_reason, str):
            raise TypeError("failure_reason must be a string or None")

        if self.success:
            if (
                self.structure_path is None
                or self.mapping is None
                or self.manipulator is None
            ):
                raise ValueError(
                    "successful evaluation requires a structure path, mapping, "
                    "and manipulator"
                )
            if self.failure_reason is not None:
                raise ValueError(
                    "successful evaluation must not include a failure reason"
                )
        else:
            if self.manipulator is not None:
                raise ValueError("failed evaluation must not include a manipulator")
            if not self.failure_reason:
                raise ValueError("failed evaluation requires a failure reason")

        object.__setattr__(self, "input_index", int(self.input_index))
        object.__setattr__(self, "energy", normalized_energy)


class ExplicitOwnershipEvaluator:
    """Adapt evaluator callbacks to explicit-ownership candidate results.

    :param GB: Keyword argument, required. Reference grain-boundary construction.
    :param scalar_energy_func: Keyword argument, required. Scalar evaluator callback.
    :param batch_energy_func: Keyword argument, required. Ordered batch evaluator
        callback, or ``None``.
    :param local_random: Keyword argument, required. Optimizer-owned random-number
        generator.
    :param penalty: Keyword argument, required. Optimizer-supplied finite energy used
        for failed calculations.
    :raises TypeError: If a callback, random-number generator, or penalty has an
        invalid type.
    :raises ValueError: If ``penalty`` is non-finite.
    """

    def __init__(
        self,
        *,
        GB: GBMaker,
        scalar_energy_func: Callable,
        batch_energy_func: Callable | None,
        local_random: np.random.Generator,
        penalty: float,
    ) -> None:
        """Initialize the explicit-ownership evaluator adapter.

        :param GB: Keyword argument, required. Reference grain-boundary construction.
        :param scalar_energy_func: Keyword argument, required. Scalar evaluator
            callback.
        :param batch_energy_func: Keyword argument, required. Ordered batch callback, or
            ``None``.
        :param local_random: Keyword argument, required. Optimizer-owned random-number
            generator.
        :param penalty: Keyword argument, required. Optimizer-supplied finite energy
            used for failed calculations.
        :raises TypeError: If a callback, random-number generator, or penalty has an
            invalid type.
        :raises ValueError: If ``penalty`` is non-finite.
        """
        if not callable(scalar_energy_func):
            raise TypeError("scalar_energy_func must be callable")
        if batch_energy_func is not None and not callable(batch_energy_func):
            raise TypeError("batch_energy_func must be callable or None")
        if not isinstance(local_random, np.random.Generator):
            raise TypeError("local_random must be a numpy.random.Generator")
        if isinstance(penalty, (bool, np.bool_)) or not isinstance(penalty, Real):
            raise TypeError("penalty must be a non-Boolean real scalar")
        normalized_penalty = float(penalty)
        if not np.isfinite(normalized_penalty):
            raise ValueError("penalty must be finite")

        self.GB = GB
        self.scalar_energy_func = scalar_energy_func
        self.batch_energy_func = batch_energy_func
        self.local_random = local_random
        self.penalty = normalized_penalty
        self._claimed_paths: set[Path] = set()

    def begin_run(self) -> None:
        """Reset run-local evaluator artifact identity tracking."""
        self._claimed_paths.clear()

    def _candidate_file_mapping(
        self,
        manipulator: GBManipulator,
        atoms: np.ndarray,
    ) -> CandidateFileMapping:
        """Build the transient file mapping for one candidate.

        :param manipulator: Candidate manipulator carrying persistent labels.
        :param atoms: Candidate atom rows aligned with those labels.
        :return: Validated candidate/file mapping.
        :raises GrainOwnershipError: If ownership did not propagate or geometry is
            inconsistent.
        """
        labels = manipulator.candidate_grain_labels
        if labels is None:
            raise GrainOwnershipError(
                "explicit-ownership mutation did not propagate grain labels"
            )
        parent = manipulator.parents[0]
        return CandidateFileMapping.from_candidate(
            atoms,
            labels,
            box_dims=parent.box_dims,
            gb_plane_x=parent.gb_plane_x,
            inplane_periodic=parent.inplane_periodic,
            left_grain_x_bounds=parent.left_grain_x_bounds,
            right_grain_x_bounds=parent.right_grain_x_bounds,
            coordinate_tolerance=parent.coordinate_tolerance,
            normal_topology=parent.normal_topology,
        )

    def _failed_evaluation(
        self,
        input_index: int,
        reason: str,
        mapping: CandidateFileMapping | None = None,
        structure_path: str | None = None,
    ) -> CandidateEvaluation:
        """Create one penalty-bearing failed evaluation result.

        :param input_index: Candidate position in the submitted population, or ``-1``
            for the initial seed evaluation.
        :param reason: Human-readable failure context.
        :param mapping: Optional, defaults to ``None``. Candidate/file mapping when
            construction reached that stage.
        :param structure_path: Optional, defaults to ``None``. Canonical artifact path
            when supplied by the evaluator.
        :return: Failed evaluation carrying both penalty and failure context.
        """
        return CandidateEvaluation(
            input_index=input_index,
            energy=self.penalty,
            structure_path=structure_path,
            mapping=mapping,
            manipulator=None,
            success=False,
            failure_reason=reason,
        )

    @staticmethod
    def _normalize_energy(energy: object) -> float:
        """Normalize one evaluator energy.

        :param energy: Evaluator-returned energy value.
        :return: Finite Python float.
        :raises TypeError: If the value is Boolean or is not a real scalar.
        :raises ValueError: If the value is non-finite.
        """
        if isinstance(energy, (bool, np.bool_)) or not isinstance(energy, Real):
            raise TypeError("energy must be a non-Boolean real scalar")
        normalized = float(energy)
        if not np.isfinite(normalized):
            raise ValueError("energy must be finite")
        return normalized

    @staticmethod
    def _diagnostic_path(structure_path: object) -> str | None:
        """Return a canonical diagnostic path when one was supplied.

        :param structure_path: Evaluator-returned path-like value.
        :return: Canonical path string, or None for a non-path value.
        :raises OSError: If the path cannot be resolved.
        :raises RuntimeError: If path resolution encounters a symlink loop.
        :raises ValueError: If the path contains an invalid value.
        """
        if not isinstance(structure_path, (str, Path)):
            return None
        return str(Path(structure_path).resolve())

    def _reload_mapping(
        self,
        structure_path: str,
        mapping: CandidateFileMapping,
    ) -> GBManipulator:
        """Validate and reconstruct one evaluator artifact.

        :param structure_path: Evaluator-returned structure path.
        :param mapping: Expected candidate ownership and geometry.
        :return: Reconstructed manipulator with the optimizer RNG attached.
        :raises FileNotFoundError: If the artifact does not exist.
        :raises LammpsDataError: If the artifact cannot be read unambiguously.
        :raises GrainOwnershipError: If the artifact changed candidate identity or
            could not reconstruct a valid candidate.
        """
        try:
            manipulator = reload_explicit_manipulator(
                structure_path,
                candidate_mapping=mapping,
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
                type_dict=self.GB.unit_cell.type_map,
            )
        except (ParentError, GBManipulatorError) as exc:
            raise GrainOwnershipError(
                f"evaluator artifact could not reconstruct a valid candidate: {exc}"
            ) from exc
        manipulator.rng = self.local_random
        return manipulator

    def _record_result(
        self,
        *,
        input_index: int,
        mapping: CandidateFileMapping,
        energy: object = _MISSING,
        structure_path: object = _MISSING,
    ) -> CandidateEvaluation:
        """Normalize and validate one callback result.

        Missing energy or structure output is treated as a failed calculation and
        receives the optimizer penalty. Differentiating structural failures from other
        evaluator failures is intentionally deferred until evaluators expose a typed
        failure classification.

        :param input_index: Keyword argument, required. Candidate position in the
            submitted population, or ``-1`` for the initial seed evaluation.
        :param mapping: Keyword argument, required. Candidate/file mapping
            established before evaluation.
        :param energy: Keyword argument, optional, defaults to ``_MISSING``.
            Evaluator-returned energy, or an internal missing sentinel.
        :param structure_path: Keyword argument, optional, defaults to ``_MISSING``.
            Evaluator-returned artifact path, or an internal missing sentinel.
        :return: Successful reconstructed evaluation or a penalty-bearing failure.
        """
        missing_fields = []
        if energy is _MISSING or energy is None:
            missing_fields.append("energy")
        if structure_path is _MISSING or structure_path is None:
            missing_fields.append("final_dump")
        try:
            diagnostic_path = self._diagnostic_path(structure_path)
        except (OSError, RuntimeError, ValueError) as exc:
            return self._failed_evaluation(
                input_index,
                f"invalid structure path: {type(exc).__name__}: {exc}",
                mapping,
            )
        if missing_fields:
            return self._failed_evaluation(
                input_index,
                "incomplete evaluator result missing " + ", ".join(missing_fields),
                mapping,
                diagnostic_path,
            )

        try:
            numeric_energy = self._normalize_energy(energy)
        except (TypeError, ValueError) as exc:
            return self._failed_evaluation(
                input_index,
                f"invalid energy: {exc}",
                mapping,
                diagnostic_path,
            )

        if diagnostic_path is None:
            return self._failed_evaluation(
                input_index,
                "evaluator did not return a structure path",
                mapping,
            )

        path = Path(diagnostic_path)
        if not path.is_file():
            return self._failed_evaluation(
                input_index,
                "evaluator did not return a valid structure path",
                mapping,
                diagnostic_path,
            )
        if path in self._claimed_paths:
            return self._failed_evaluation(
                input_index,
                f"evaluator reused a structure path already assigned in this run: {path}",
                mapping,
                diagnostic_path,
            )

        try:
            # pyraisecontract: ignore=DOC115[FileNotFoundError]
            #   FileNotFoundError is a subclass of OSError and is converted to a
            #   failed evaluation by the surrounding OSError recovery boundary.
            manipulator = self._reload_mapping(diagnostic_path, mapping)
        except (OSError, LammpsDataError, GrainOwnershipError) as exc:
            return self._failed_evaluation(
                input_index,
                f"{type(exc).__name__}: {exc}",
                mapping,
                diagnostic_path,
            )

        self._claimed_paths.add(path)
        return CandidateEvaluation(
            input_index=input_index,
            energy=numeric_energy,
            structure_path=diagnostic_path,
            mapping=mapping,
            manipulator=manipulator,
            success=True,
        )

    def evaluate_candidate(
        self,
        manipulator: GBManipulator,
        atoms: np.ndarray,
        unique_id: str,
        input_index: int,
    ) -> CandidateEvaluation:
        """Evaluate and reconstruct one candidate.

        :param manipulator: Candidate manipulator carrying ownership state.
        :param atoms: Candidate atom rows.
        :param unique_id: Evaluator invocation identifier.
        :param input_index: Candidate position in the population, or ``-1`` for the
            initial seed evaluation.
        :return: Normalized candidate evaluation.
        """
        try:
            mapping = self._candidate_file_mapping(manipulator, atoms)
        except GrainOwnershipError as exc:
            return self._failed_evaluation(input_index, str(exc))

        try:
            result = self.scalar_energy_func(
                self.GB,
                manipulator,
                atoms,
                unique_id,
            )
            energy, structure_path = result
        except Exception as exc:
            # The external evaluator callback is a deliberate recovery boundary.
            return self._failed_evaluation(
                input_index,
                f"{type(exc).__name__}: {exc}",
                mapping,
            )

        return self._record_result(
            input_index=input_index,
            mapping=mapping,
            energy=energy,
            structure_path=structure_path,
        )

    def evaluate_generation(
        self,
        population_manipulators: list[GBManipulator],
        population_structures: list[np.ndarray],
        population_lineages: list[list[str]],
        gen: int,
        unique_id: int | UUID,
    ) -> list[CandidateEvaluation]:
        """Evaluate one index-aligned explicit-ownership population.

        :param population_manipulators: Candidate manipulators in population order.
        :param population_structures: Candidate atom arrays in population order.
        :param population_lineages: Candidate lineages in population order.
        :param gen: Generation index.
        :param unique_id: Run identifier used to construct callback IDs.
        :return: One aligned typed evaluation per input candidate.
        :raises ValueError: If population arrays or batch results are not aligned, or if
            a batch result is not a dictionary.
        :raises RuntimeError: If an internal alignment invariant is lost.
        """
        population_length = len(population_structures)
        if not (
            len(population_manipulators)
            == len(population_lineages)
            == population_length
        ):
            raise ValueError(
                "explicit-ownership population manipulators, structures, and lineages "
                "must remain index-aligned"
            )

        unique_ids = [
            f"GA_{unique_id}_g{gen}_c{i}" for i in range(population_length)
        ]
        if self.batch_energy_func is None:
            return [
                self.evaluate_candidate(manipulator, atoms, candidate_id, index)
                for index, (manipulator, atoms, candidate_id) in enumerate(
                    zip(population_manipulators, population_structures, unique_ids)
                )
            ]

        records: list[CandidateEvaluation | None] = [None] * population_length
        valid_indices: list[int] = []
        valid_mappings: list[CandidateFileMapping] = []
        for index, (manipulator, atoms) in enumerate(
            zip(population_manipulators, population_structures)
        ):
            try:
                mapping = self._candidate_file_mapping(manipulator, atoms)
            except GrainOwnershipError as exc:
                records[index] = self._failed_evaluation(index, str(exc))
                continue
            valid_indices.append(index)
            valid_mappings.append(mapping)

        if valid_indices:
            try:
                raw_results = self.batch_energy_func(
                    self.GB,
                    [population_manipulators[index] for index in valid_indices],
                    [population_structures[index] for index in valid_indices],
                    [population_lineages[index] for index in valid_indices],
                    [unique_ids[index] for index in valid_indices],
                )
            except Exception as exc:
                # The external evaluator callback is a deliberate recovery boundary.
                for input_index, mapping in zip(valid_indices, valid_mappings):
                    records[input_index] = self._failed_evaluation(
                        input_index,
                        f"{type(exc).__name__}: {exc}",
                        mapping,
                    )
            else:
                if not isinstance(raw_results, list) or len(raw_results) != len(
                    valid_mappings
                ):
                    raise ValueError(
                        "explicit-ownership batch evaluation requires one ordered "
                        "result dictionary per input candidate"
                    )
                for result_index, result in enumerate(raw_results):
                    if not isinstance(result, dict):
                        raise ValueError(
                            f"explicit-ownership batch result {result_index} must be a "
                            "dictionary"
                        )
                for input_index, mapping, result in zip(
                    valid_indices,
                    valid_mappings,
                    raw_results,
                ):
                    records[input_index] = self._record_result(
                        input_index=input_index,
                        mapping=mapping,
                        energy=result.get("energy", _MISSING),
                        structure_path=result.get("final_dump", _MISSING),
                    )

        if any(record is None for record in records):
            raise RuntimeError("explicit-ownership evaluation lost candidate alignment")
        return [record for record in records if record is not None]
