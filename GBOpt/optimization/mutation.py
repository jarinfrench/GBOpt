# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Random mutation selection over GBManipulator operations."""

import numpy as np

from GBOpt import GBMaker, GBManipulator
from GBOpt.GBManipulator import GBManipulatorValueError
from GBOpt.optimization.types import GBMinimizerError, GBMinimizerValueError


class Mutator:
    """Perform randomly selected manipulations on a GB candidate.

    :param choices: Mutation operation names to make available.
    :param manipulator: GBManipulator used to validate the requested operations.
    """

    # TODO: Add more manipulator options to this class as we make more
    # manipulators faster.

    def __init__(self, choices: list[str], manipulator: GBManipulator):
        invalid_choices = [
            method for method in choices if not hasattr(manipulator, method)
        ]
        if invalid_choices:
            raise GBMinimizerValueError(
                "Unknown GBManipulator mutation choice(s): "
                + ", ".join(repr(choice) for choice in invalid_choices)
            )

        # Duplicate names do not weight a mutation more heavily.
        self.choices_keys = list(dict.fromkeys(choices))
        if not self.choices_keys:
            raise GBMinimizerValueError(
                "At least one mutation choice must be provided."
            )

    def _apply_mutation(
        self,
        choice_key: str,
        *,
        local_random: np.random.Generator,
        GB: GBMaker,
        manipulator: GBManipulator,
    ):
        """Apply one explicitly selected mutation.

        :param choice_key: Mutation operation to apply.
        :param local_random: Optimizer-owned random-number generator.
        :param GB: GBMaker providing boundary dimensions and repeat factors.
        :param manipulator: GBManipulator on which to perform the mutation.
        :return: Mutation description and resulting atom positions.
        :raises GBManipulatorValueError: If the selected mutation is infeasible.
        :raises GBMinimizerValueError: If ``choice_key`` is unsupported.
        """
        match choice_key:
            case "insert_atoms":
                new_system = manipulator.insert_atoms(
                    method="grid",
                    num_to_insert=1,
                )
                mutation = "add1"

            case "remove_atoms":
                new_system = manipulator.remove_atoms(num_to_remove=1)
                mutation = "remove1"

            case "translate_right_grain":
                parent = manipulator.parents[0]
                y_dim = parent.box_dims[1, 1] - parent.box_dims[1, 0]
                z_dim = parent.box_dims[2, 1] - parent.box_dims[2, 0]

                dy = (y_dim / GB.repeat_factor[0]) * local_random.uniform(0, 1)
                dz = (z_dim / GB.repeat_factor[1]) * local_random.uniform(0, 1)

                new_system = manipulator.translate_right_grain(dy=dy, dz=dz)
                mutation = f"shift{dy:.8f}dy{dz:.8f}dz"

            case _:
                raise GBMinimizerValueError(
                    f"Unhandled mutation choice: {choice_key!r}"
                )

        return mutation, new_system

    def mutate(
        self,
        local_random: np.random.Generator,
        GB: GBMaker,
        manipulator: GBManipulator,
    ):
        """Perform a randomly selected feasible mutation.

        Each configured mutation is attempted at most once. If an operation is
        physically infeasible for the current candidate, another configured operation is
        tried. Failure of every configured operation is fatal.

        :param local_random: Optimizer-owned random-number generator.
        :param GB: GBMaker providing boundary dimensions and repeat factors.
        :param manipulator: GBManipulator on which to perform the mutation.
        :return: Mutation description and resulting atom positions.
        :raises GBMinimizerError: If no configured mutation can produce a candidate.
        """
        choice_order = local_random.permutation(len(self.choices_keys))
        failures: list[tuple[str, GBManipulatorValueError]] = []

        for choice_index in choice_order:
            choice_key = self.choices_keys[int(choice_index)]
            try:
                return self._apply_mutation(
                    choice_key,
                    local_random=local_random,
                    GB=GB,
                    manipulator=manipulator,
                )
            except GBManipulatorValueError as exc:
                failures.append((choice_key, exc))

        failure_details = "; ".join(
            f"{choice}: {exc}" for choice, exc in failures
        )
        error = GBMinimizerError(
            "No configured mutation could produce a valid candidate. Attempted "
            f"mutations: {failure_details}"
        )
        raise error from failures[-1][1]

