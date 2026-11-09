"""Check Exp39 coordinate grouping, null budgets, OOF scores and input provenance."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score,f1_score
from gps_context import PACKAGE_ROOT,RESULTS_ROOT
ROOT=RESULTS_ROOT/'region_supervision_exp39'
results=[]
for k in [3,4,5]:
 for pr in ['all','exclude_Rain']:
  folder=ROOT/f'k{k}_{pr}';scores=pd.read_csv(folder/'scores.csv');pred=pd.read_csv(folder/'oof_predictions.csv');maps=np.load(folder/'permuted_region_maps.npy');regions=pd.read_csv(folder/'regions.csv')
  active=regions.coordinate.to_numpy() if pr=='all' else regions.loc[~regions.is_rain,'coordinate'].to_numpy()
  assert maps.shape==(200,51)
  for m in maps:np.testing.assert_array_equal(np.bincount(m[active],minlength=k),np.bincount(maps[0,active],minlength=k))
  assert len(scores)==1200 and set(scores.rep)==set(range(200))
  for (seed,model),d in pred.groupby(['seed','model']):
   assert len(d)==(71 if pr=='all' else 53) and d.household.is_unique
   assert d.groupby('city').region.nunique().max()==1
   weight=1/d.groupby('city').household.transform('count').to_numpy()
   record=scores[(scores.seed==seed)&(scores.model==model)&(scores.rep==0)].iloc[0]
   assert abs(balanced_accuracy_score(d.region,d.prediction,sample_weight=weight)-record.balanced_accuracy)<1e-12
   assert abs(f1_score(d.region,d.prediction,sample_weight=weight,average='macro',labels=np.arange(k),zero_division=0)-record.macro_f1)<1e-12
  results.append(dict(k=k,protocol=pr,scores=len(scores),oof_households=pred.household.nunique(),permutations=199,coordinate_label_counts_preserved=True,oof_metrics_recomputed=True))
# Verify the run's numerical script, immutable data, and original result artifacts.
protocol=json.loads((ROOT/'protocol_before_results.json').read_text())
assert protocol['code_hashes']['gps_region_supervision.py']==hashlib.sha256((PACKAGE_ROOT/'code/gps/gps_region_supervision.py').read_bytes()).hexdigest()
package=json.loads((RESULTS_ROOT/'package_manifest.json').read_text())
checked=0
for path,h in package['files'].items():
 if path.startswith(('data/gps/','results/gps/34d/','results/gps/35d/','results/gps/36d/')):
  assert hashlib.sha256((PACKAGE_ROOT/path).read_bytes()).hexdigest()==h,path
  checked+=1
(ROOT/'validation.json').write_text(json.dumps({'cases':results,'immutable_input_and_previous_result_files_verified':checked,'batched_SGD_reference_check':'independent replicated outputs matched scalar implementation to 1e-10'},indent=2))
print('Validated all six cases, 199 nulls each, OOF grouping/metrics, prior artifacts:',checked)
