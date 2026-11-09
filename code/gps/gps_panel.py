"""Explicit GPS input-space and output-root configuration; legacy defaults stay 34D.

ENERGY_GPS_INPUT_DIMS=36 retains all policy coordinates, including index 3
(location weight) and index 35 (bias). ENERGY_GPS_RESULT_ROOT isolates reruns.
Neither setting changes the 36D policy used for behavioral replay.
"""
import os
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from gps_context import tail, RESULTS_ROOT


def input_indices():
    dimension = int(os.environ.get('ENERGY_GPS_INPUT_DIMS', '34'))
    if dimension not in (34, 35, 36):
        raise ValueError('ENERGY_GPS_INPUT_DIMS must be 34, 35 or 36')
    if dimension == 35:
        return [j for j in range(36) if j != 3]
    return list(range(36)) if dimension == 36 else tail.paths.dims_geometry(36)


def result_root():
    return Path(os.environ.get('ENERGY_GPS_RESULT_ROOT', str(RESULTS_ROOT/f'{len(input_indices())}d'))).resolve()


def load_panel():
    root = Path(tail.paths.SEED_ROOT)
    seeds = sorted(p.name for p in root.iterdir() if p.is_dir() and list(p.glob('*_expert.npy')))
    ids = None
    raw = {}
    keep = input_indices()
    for seed in seeds:
        files = sorted((root / seed).glob('*_expert.npy'))
        current = [p.name.removesuffix('_expert.npy') for p in files]
        if ids is None:
            ids = current
        if current != ids:
            raise ValueError(f'Household IDs differ: {seed}')
        weights = np.stack([np.load(p) for p in files])
        if weights.shape != (71, 36) or not np.isfinite(weights).all():
            raise ValueError(f'Invalid policy panel: {seed}')
        raw[seed] = weights[:, keep]
    scaler = StandardScaler().fit(np.vstack(list(raw.values())))
    return ids, seeds, raw, {s: scaler.transform(raw[s]) for s in seeds}, scaler, keep
