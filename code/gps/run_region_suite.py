"""Run preregistered Exp39 region supervision grid, with complete null budgets."""
import concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1];OUT=PACKAGE/'results/gps/region_supervision_exp39'
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir();(OUT/'logs').mkdir()
config={'primary':{'k':4,'protocol':'all','input_dims':35,'model':'triplet_mlp','metric':'coordinate_balanced_OOF_balanced_accuracy'},'sensitivity_k':[3,5],'protocols':['all','exclude_Rain'],'region_definition':'KMeans on unit-sphere xyz at 51 unique coordinates, n_init=50, random_state=0; same regions for both protocols','min_region_households':8,'folds':'reuse Exp36 fixed 5 coordinate-group folds','permutations':199,'null':'shuffle coordinate-region mapping within protocol, preserve region coordinate counts; same coordinate keeps same label','architecture':[35,64,64,64,64],'relu_hidden_layers':3,'epochs':10,'triplets_per_fit':256,'batch_size':32,'SGD_lr':0.01,'margin':1.,'positive':'same region, distinct coordinates; first sample coordinates, then households','classifier':'nearest region centroid, training coordinates weighted equally','baseline':'same classifier on standardized raw theta','selection':'no hyperparameter selection on OOF; K fixed before scores','multiplicity':'all six configs and nulls reported; Holm correction over six MLP BA tests and separately six paired-gain tests','full_panel_cluster_analysis':'HDBSCAN and KMeans at region K; descriptive, no cluster null','code_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
(OUT/'protocol_before_results.json').write_text(json.dumps(config,indent=2))
environment=os.environ.copy();environment.update(ENERGY_GPS_INPUT_DIMS='35',MPLCONFIGDIR='/private/tmp/energy-gps-mpl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
def run(k,pr):
 start=time.monotonic()
 with (OUT/'logs'/f'k{k}_{pr}.log').open('w') as f:
  p=subprocess.run([sys.executable,str(HERE/'gps_region_supervision.py'),'--k',str(k),'--protocol',pr],cwd=PACKAGE,env=environment,stdout=f,stderr=subprocess.STDOUT)
 r={'k':k,'protocol':pr,'returncode':p.returncode,'seconds':time.monotonic()-start};print(json.dumps(r),flush=True);return r
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 tasks=[pool.submit(run,k,pr) for k in [4,3,5] for pr in ['all','exclude_Rain']]
 for t in concurrent.futures.as_completed(tasks):
  results.append(t.result());(OUT/'status.json').write_text(json.dumps(results,indent=2))
if any(r['returncode'] for r in results):raise RuntimeError('Failed case: inspect logs')
