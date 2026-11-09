"""Run complete GPS protocols, with 34/35/36 input dimensions and no overwrites."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
EXPERIMENTS=['gps_triplet_pilot.py','gps_cluster_comparison.py','gps_network_discovery.py',
             'gps_place_networks.py','gps_distance_regression.py','gps_behavior_increment.py']


def main(args):
    out=(args.output or PACKAGE/f'results/gps/{args.dims}d').resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}; choose a new --output')
    keep=[i for i in range(36) if i not in ([3,35] if args.dims==34 else [3] if args.dims==35 else [])]
    out.mkdir(parents=True);(out/'logs').mkdir()
    source=list(HERE.glob('*.py'))+[p for p in (PACKAGE/'data/gps').rglob('*') if p.is_file()]
    config=dict(input_dimension=args.dims,retained_original_indices=keep,dropped_indices=sorted(set(range(36))-set(keep)),
                bias_mode='original_bias',location_folded_into_bias=False,cem_retrained=False,n_households=71,cem_seeds=[0,1,42],
                permutation_budgets={'exp33':99,'exp34':39,'exp35':39,'exp36':99},python=sys.executable,
                input_sha256={str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    (out/'run_manifest.json').write_text(json.dumps(config,indent=2))
    environment=os.environ.copy();environment.update(ENERGY_GPS_INPUT_DIMS=str(args.dims),ENERGY_GPS_RESULT_ROOT=str(out),
        MPLCONFIGDIR=str(Path(os.environ.get('TMPDIR','/tmp'))/'energy-gps-mpl'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def run(name):
        start=time.monotonic()
        with (out/'logs'/name.replace('.py','.log')).open('w') as f:
            p=subprocess.run([sys.executable,str(HERE/name)],cwd=PACKAGE,env=environment,stdout=f,stderr=subprocess.STDOUT)
        result=dict(script=name,returncode=p.returncode,seconds=time.monotonic()-start)
        print(json.dumps(result),flush=True)
        return result
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(run,name) for name in EXPERIMENTS]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result());(out/'status.json').write_text(json.dumps(results,indent=2))
    if any(r['returncode'] for r in results):
        raise RuntimeError('A numerical experiment failed; inspect logs.')
    for name in ['summarize_gps_network.py','gps_spectral_comparison.py','plot_gps_distance_clusters.py',
                 'plot_place_clusters.py','audit_candidates.py']:
        r=run(name);results.append(r);(out/'status.json').write_text(json.dumps(results,indent=2))
        if r['returncode']:
            raise RuntimeError(f'Postprocessing failed: {name}')
    print(f'Complete {args.dims}D suite: {out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dims',type=int,choices=[34,35,36],required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    if args.jobs<1:
        parser.error('--jobs must be positive')
    main(args)
