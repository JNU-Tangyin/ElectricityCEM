"""Portable package paths and adapters for the unchanged numerical kernels."""
from pathlib import Path
from types import SimpleNamespace
import energy_storage_env as env
from observation_matrix import build_obs_matrix

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PACKAGE_ROOT / 'data/gps'
RESULTS_ROOT = PACKAGE_ROOT / 'results/gps'
WINDOW_H = 72

# Compatibility namespace for the existing experiment implementations.
# This module has no dependency on the project's shape directory.
paths = SimpleNamespace(ROOT=PACKAGE_ROOT, SEED_ROOT=INPUT_ROOT/'policies',
    FEATURES_CSV=INPUT_ROOT/'household_features_full.csv', META_PATH=INPUT_ROOT/'german_metadata.json',
    SHAPE=RESULTS_ROOT/'34d', dims_geometry=lambda n:[j for j in range(n) if j not in (3,35)])
tail = SimpleNamespace(paths=paths, DATA_DIR=INPUT_ROOT/'households', SELL_RATIO=1.0)
