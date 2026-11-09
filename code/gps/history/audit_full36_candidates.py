"""Repeat the minority audit with 36D labels and replay all saved checkpoints.

Data descriptors are invariant to the representation dimension and are reused
only after verifying all original data/policy hashes from the previous audit.
Membership, reference groups, distances, ranks and checkpoint replay are recomputed.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/energy-mpl-cache')
from pathlib import Path
import hashlib
import json
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits
from gps_panel import load_panel, result_root, input_indices
from tail_action_replay import tail, env
from audit_raw_small_cluster import rollout, markdown


def main():
    assert len(input_indices())==36
    root=Path(tail.paths.ROOT)
    old=root/'04.results/shape/exp38_raw_small_cluster_audit'
    out=result_root()/'exp38_raw_small_cluster_audit'
    if out.exists():
        raise FileExistsError(out)
    manifest=json.loads((old/'manifest.json').read_text())
    checked=[]
    for item in manifest['files']:
        if item['path'].startswith('00.data/') or '_expert.npy' in item['path'] or '_history.csv' in item['path']:
            path=root/item['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
            checked.append(item)
    data=pd.read_csv(old/'all_households_data.csv',dtype={'household':str})
    params=pd.read_csv(old/'all_households_parameters_training.csv',dtype={'household':str})
    data=data.drop(columns=[c for c in data if '_pct_reference61' in c])
    params=params.drop(columns=[c for c in params if '_pct_reference61' in c])
    ids,seeds,raw,*_=load_panel()
    saved=pd.read_csv(result_root()/'exp34_gps_network_discovery/labels.csv',dtype={'source':str})
    small={};labs={};coords={}
    for s in seeds:
        z=StandardScaler().fit_transform(raw[s]);coords[s]=z
        labels=HDBSCAN(min_cluster_size=4,min_samples=2).fit_predict(z)
        assert np.array_equal(labels,saved[(saved.seed==s)&(saved.model=='raw')].set_index('source').loc[ids,'cluster'])
        labs[s]=labels
        real=set(labels)-{-1}
        small[s]=set(np.array(ids)[labels==min(real,key=lambda v:sum(labels==v))]) if len(real)>=2 else set()
    cand=set.union(*small.values());core=set.intersection(*small.values())
    group=lambda h:'core' if h in core else 'other_candidates' if h in cand else 'reference'
    for df in [data,params]:
        df['candidate']=df.household.isin(cand);df['core']=df.household.isin(core);df['group']=df.household.map(group)
    data['minority_seed_count']=data.household.map(lambda h:sum(h in small[s] for s in seeds))
    for s in seeds:
        idx=params.index[params.seed==s]
        h=params.loc[idx,'household'].tolist()
        assert h==ids
        params.loc[idx,'cluster']=labs[s]
        params.loc[idx,'in_small_cluster']=[x in small[s] for x in h]
        params.loc[idx,'standardized_centroid_distance']=np.linalg.norm(coords[s]-coords[s].mean(axis=0),axis=1)
        params.loc[idx,'full36_norm']=np.linalg.norm(raw[s],axis=1)
    for i,h in enumerate(ids):
        frame=pd.read_csv(Path(tail.DATA_DIR)/f'{h}.csv',index_col=0,parse_dates=True)
        train=env.EnergyStorageEnv(frame.iloc[:-96].copy(),episode_length_hours=72,sell_price_ratio=tail.SELL_RATIO)
        val=env.EnergyStorageEnv(frame.iloc[-96:].copy(),episode_length_hours=95,sell_price_ratio=tail.SELL_RATIO)
        for s in seeds:
            w=raw[s][i]
            ix=params.index[(params.household==h)&(params.seed==s)][0]
            params.loc[ix,'replay_checkpoint_train_fixed3_eur_h']=rollout(train,w,[0,59,118])
            params.loc[ix,'replay_checkpoint_val_eur_h']=rollout(val,w,[0])
            params.loc[ix,'val_replay_abs_error']=abs(params.loc[ix,'replay_checkpoint_val_eur_h']-params.loc[ix,'logged_best_val_profit_eur_h'])
        train.close();val.close()
    assert params.val_replay_abs_error.max()<1e-9
    fields=['frac_at_bound','max_abs_weight','w_norm','bias','standardized_centroid_distance','best_iter']
    for s in seeds:
        mask=params.seed==s
        for metric in fields:
            ref=params.loc[mask & ~params.candidate,metric].to_numpy()
            params.loc[mask,metric+'_pct_reference']=params.loc[mask,metric].map(lambda v:100*((ref<v).mean()+.5*(ref==v).mean()))
    joined=params.merge(data.drop(columns=['group','core','candidate']),on='household',validate='many_to_one')
    overlap=pd.DataFrame([dict(seed_a=a,seed_b=b,intersection=len(small[a]&small[b]),union=len(small[a]|small[b]),jaccard=len(small[a]&small[b])/len(small[a]|small[b]) if small[a]|small[b] else np.nan) for a,b in combinations(seeds,2)])
    membership=pd.DataFrame([dict(seed=s,household=h,in_small_cluster=h in small[s],cluster=int(labs[s][i])) for s in seeds for i,h in enumerate(ids)])
    old_candidates=set(pd.read_csv(old/'all_households_data.csv',dtype={'household':str}).query('candidate').household)
    out.mkdir()
    for name,df in [('all_households_data',data),('all_households_parameters_training',params),('candidate_audit',joined[joined.candidate]),('core_audit',joined[joined.core]),('small_cluster_overlap',overlap),('membership',membership)]:
        df.to_csv(out/f'{name}.csv',index=False)
    summary={'input_dims':36,'candidate_n':len(cand),'core_n':len(core),'reference_n':71-len(cand),
             'candidates':sorted(cand),'core':sorted(core),'small_by_seed':{s:sorted(v) for s,v in small.items()},
             'candidate_intersection_with_34d':sorted(cand&old_candidates),
             'max_abs_weight':float(params.max_abs_weight.max()),'val_replay_max_error':float(params.val_replay_abs_error.max()),
             'data_descriptor_source':str(old/'all_households_data.csv'),'verified_invariant_input_hashes':checked,
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'audit.json').write_text(json.dumps(summary,indent=2))
    display=data[data.candidate][['household','minority_seed_count','city','zero_load_fraction','pv_peak_wh','complete_day_load_mean_kwh']]
    (out/'REPORT.md').write_text(f'# 完整36维原始θ小簇检查\n\n小簇并集{len(cand)}户，共同成员{len(core)}户：'+ '、'.join(sorted(core))+'。\n\n'+markdown(display)+'\n\n'+markdown(overlap)+f'\n\n全部213个checkpoint重新执行原家庭训练/验证回放，验证日志最大误差{params.val_replay_abs_error.max():.3g} EUR/h。参数/输入数据指标沿用相同原始文件，已验证输入哈希；36维簇标签、候选组、质心距离和对照百分位重新计算。\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='verified_invariant_input_hashes'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
