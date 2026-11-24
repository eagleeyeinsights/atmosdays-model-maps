import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.train import FEATURES, load


class TrainingDataTests(unittest.TestCase):
    def test_only_annual_training_chunks_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            for cache in ("live-2026-11", "obs-2026", "gfs-2026-extra"):
                np.savez(data / cache, cached=[42])
            for name, rows in (("rf-2019", [2]), ("gfs-2026", [1]), ("rf-2018", [])):
                np.savez(data / name, features=FEATURES, X=rows, Y=rows, M=rows)
            for array in load(data):
                np.testing.assert_array_equal(array, [1, 2])
