# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import logging
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from GBOpt.GBMaker import GBMaker
from GBOpt.GBMinimizer import MonteCarloMinimizer
from GBOpt.Utils.logging_utils import get_logger


class _RecordCollector(logging.Handler):

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class TestMonteCarloMinimizerLogging(unittest.TestCase):

    def setUp(self):
        theta = math.radians(36.869898)
        misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.gb = GBMaker(
            3.52,
            "fcc",
            10.0,
            misorientation,
            "Ni",
            repeat_factor=2,
            x_dim_min=30.0,
            vacuum=8.0,
            interaction_distance=8.0,
        )
        self.tmpdir = tempfile.TemporaryDirectory()
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        self.logger = get_logger("GBOpt.GBMinimizer")
        self.handler = _RecordCollector()
        self.handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)
        self.original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.original_level)
        os.chdir(self.orig_cwd)
        self.tmpdir.cleanup()

    def _make_minimizer(self, energy_func):
        return MonteCarloMinimizer(
            self.gb,
            energy_func,
            ["translate_right_grain"],
            seed=0,
        )

    def test_run_mc_logs_energy_tolerance_event(self):
        def energy_func(GB, manipulator, atom_positions, unique_id):
            dump_file = f"{unique_id}.data"
            GB.write_lammps(
                dump_file,
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            return 1.0, dump_file

        mc = self._make_minimizer(energy_func)
        energy = mc.run_MC(max_steps=2, E_tol=1e-4, unique_id=1)

        self.assertEqual(energy, 1.0)
        event_names = [record.event for record in self.handler.records]
        self.assertIn("mc_run_started", event_names)
        self.assertIn("initial_energy_evaluated", event_names)
        self.assertIn("best_energy_updated", event_names)
        self.assertIn("energy_tolerance_met", event_names)
        self.assertIn("mc_run_completed", event_names)

    def test_run_mc_logs_rejection_threshold_event(self):
        call_count = [0]

        def energy_func(GB, manipulator, atom_positions, unique_id):
            call_count[0] += 1
            dump_file = f"{unique_id}_{call_count[0]}.data"
            GB.write_lammps(
                dump_file,
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            return float(call_count[0]), dump_file

        mc = self._make_minimizer(energy_func)
        energy = mc.run_MC(max_steps=5, max_rejections=0, E_accept=1.0e-12, unique_id=2)

        self.assertEqual(energy, 1.0)
        rejection_events = [
            record for record in self.handler.records
            if getattr(record, "event", None) == "max_rejections_exceeded"
        ]
        self.assertEqual(len(rejection_events), 1)
        self.assertEqual(rejection_events[0].rejection_count, 1)
        self.assertEqual(rejection_events[0].max_rejections, 0)
