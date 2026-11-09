"""Run the full GPS protocols with all 36 theta coordinates in an isolated tree.

Usage: eenv/bin/python 02.code/multiple_country/gps/run_full36_suite.py
Default permutation budgets are unchanged: Exp33/36=99, Exp34/35=39.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
OUT = ROOT / '04.results/shape/gps_full36'
EXPERIMENTS = ['gps_triplet_pilot.py', 'gps_cluster_comparison.py',
               'gps_network_discovery.py', 'gps_place_networks.py',
               'gps_distance_regression.py', 'gps_behavior_increment.py']


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    (OUT/'logs').mkdir()
    config = dict(input_dimension=36, retained_original_indices=list(range(36)),
                  dropped_indices=[], cem_seeds=[0,1,42], n_households=71,
                  permutation_budgets={'exp33':99,'exp34':39,'exp35':39,'exp36':99},
                  unchanged='Data, folds, seeds, models, hyperparameters, replay, and null recipes',
                  cem_retrained=False, python=sys.executable, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    sources = list(SCRIPTS.glob('*.py')) + list((ROOT/'04.results/cem_npy/seed_wise').glob('cem_seed*/*_expert.npy'))
    sources += list((ROOT/'00.data/preprocessed/german_households_weather').glob('*.csv'))
    config['input_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT/'run_manifest.json').write_text(json.dumps(config,indent=2))
    env = os.environ.copy()
    env.update(ENERGY_GPS_INPUT_DIMS='36', ENERGY_GPS_RESULT_ROOT=str(OUT),
               MPLCONFIGDIR='/private/tmp/energy-mpl-cache', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    def run(name):
        start=time.monotonic()
        with (OUT/'logs'/name.replace('.py','.log')).open('w') as log:
            process=subprocess.run([sys.executable,str(SCRIPTS/name)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        result=dict(script=name,returncode=process.returncode,seconds=time.monotonic()-start)
        print(json.dumps(result),flush=True)
        return result
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,name) for name in EXPERIMENTS]
        for task in concurrent.futures.as_completed(futures):
            results.append(task.result())
            (OUT/'status.json').write_text(json.dumps(results,indent=2))
    failed=[r for r in results if r['returncode']]
    if failed:
        raise RuntimeError(f'Experiments failed; see logs: {failed}')
    for name in ['summarize_gps_network.py','gps_spectral_comparison.py',
                 'plot_gps_distance_clusters.py','plot_full36_place_clusters.py']:
        result=run(name)
        results.append(result)
        (OUT/'status.json').write_text(json.dumps(results,indent=2))
        if result['returncode']:
            raise RuntimeError(f'Postprocessing failed: {name}')
    print('Full36 numerical experiments and plots complete:',OUT,flush=True)


if __name__=='__main__':
    main()
