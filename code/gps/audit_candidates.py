"""Audit candidate households using the configured geometric input dimension."""
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
from gps_context import tail, env, INPUT_ROOT


def rollout(environment,theta,starts):
    scores=[]
    for start in starts:
        obs,_=environment.reset(options={'start_idx':int(start)})
        total=0.;steps=0;done=False
        while not done:
            action=np.tanh(obs@theta[:-1]+theta[-1])
            obs,_,done,_,info=environment.step(np.array([action]))
            total+=info['profit'];steps+=1
        scores.append(total/steps)
    return float(np.mean(scores))


def main():
    out=result_root()/'exp38_raw_small_cluster_audit'
    if out.exists():raise FileExistsError(out)
    ids,seeds,raw,*_=load_panel()
    saved=pd.read_csv(result_root()/'exp34_gps_network_discovery/labels.csv',dtype={'source':str})
    small={};labels={};standardized={}
    for s in seeds:
        z=StandardScaler().fit_transform(raw[s]);standardized[s]=z
        lab=HDBSCAN(min_cluster_size=4,min_samples=2).fit_predict(z)
        np.testing.assert_array_equal(lab,saved[(saved.seed==s)&(saved.model=='raw')].set_index('source').loc[ids,'cluster'])
        classes=set(lab)-{-1};labels[s]=lab
        small[s]=set(np.array(ids)[lab==min(classes,key=lambda c:sum(lab==c))]) if len(classes)>=2 else set()
    candidates=set.union(*small.values());core=set.intersection(*small.values())
    group=lambda h:'core' if h in core else 'other_candidates' if h in candidates else 'reference'
    coordinates=pd.read_csv(INPUT_ROOT/'coordinates_71.csv',dtype={'id':str}).set_index('id')
    data=[];parameters=[]
    for i,h in enumerate(ids):
        df=pd.read_csv(tail.DATA_DIR/f'{h}.csv',index_col=0,parse_dates=True)
        daily=df.user_load_wh.resample('D').agg(['sum','count']);daily=daily[daily['count']==24]
        loc=coordinates.loc[h]
        np.testing.assert_allclose(df[['lat','lon']].to_numpy(),np.tile([loc.latitude,loc.longitude],(len(df),1)),rtol=0,atol=1e-12)
        data.append(dict(household=h,group=group(h),candidate=h in candidates,core=h in core,
            minority_seed_count=sum(h in small[s] for s in seeds),city=loc.city,is_rain=loc.city=='Rain',
            zero_load_fraction=(df.user_load_wh==0).mean(),pv_peak_wh=df.pv_gen_wh.max(),
            complete_day_load_mean_kwh=daily['sum'].mean()/1000,
            missing_load=df.user_load_wh.isna().sum(),missing_pv=df.pv_gen_wh.isna().sum(),
            negative_load=(df.user_load_wh<0).sum(),negative_pv=(df.pv_gen_wh<0).sum()))
        train=env.EnergyStorageEnv(df.iloc[:-96].copy(),episode_length_hours=72,sell_price_ratio=1.,meta_path=str(tail.paths.META_PATH))
        val=env.EnergyStorageEnv(df.iloc[-96:].copy(),episode_length_hours=95,sell_price_ratio=1.,meta_path=str(tail.paths.META_PATH))
        for s in seeds:
            # Geometry is 34/35/36D; policy execution always uses all 36 parameters.
            w=np.load(tail.paths.SEED_ROOT/s/f'{h}_expert.npy')
            hist=pd.read_csv(tail.paths.SEED_ROOT/s/f'{h}_history.csv')
            best=int(np.argmax(hist.eval_profit.to_numpy()))
            v=rollout(val,w,[0]);t=rollout(train,w,[0,59,118])
            parameters.append(dict(seed=s,household=h,group=group(h),candidate=h in candidates,core=h in core,
                in_small_cluster=h in small[s],cluster=int(labels[s][i]),input_dims=len(input_indices()),
                max_abs_weight=np.abs(w).max(),frac_at_bound=(np.abs(w)>=1.5-1e-6).mean(),
                w_norm=np.linalg.norm(w[:-1]),bias=w[-1],standardized_centroid_distance=np.linalg.norm(standardized[s][i]-standardized[s].mean(axis=0)),
                best_iter=int(hist.iloc[best].iteration),logged_best_val_profit_eur_h=hist.iloc[best].eval_profit,
                replay_checkpoint_train_fixed3_eur_h=t,replay_checkpoint_val_eur_h=v,
                val_replay_abs_error=abs(v-hist.iloc[best].eval_profit)))
        train.close();val.close()
    data=pd.DataFrame(data);params=pd.DataFrame(parameters)
    assert params.val_replay_abs_error.max()<1e-9
    for s in seeds:
        mask=params.seed==s
        ref=params.loc[mask & ~params.candidate,'standardized_centroid_distance'].to_numpy()
        params.loc[mask,'distance_percentile_reference']=params.loc[mask,'standardized_centroid_distance'].map(lambda v:100*((ref<v).mean()+.5*(ref==v).mean()))
    joined=params.merge(data.drop(columns=['group','candidate','core']),on='household',validate='many_to_one')
    overlap=pd.DataFrame([dict(seed_a=a,seed_b=b,intersection=len(small[a]&small[b]),union=len(small[a]|small[b]),
        jaccard=len(small[a]&small[b])/len(small[a]|small[b]) if small[a]|small[b] else np.nan) for a,b in combinations(seeds,2)])
    out.mkdir()
    for name,df in [('all_households_data',data),('all_households_parameters_training',params),('candidate_audit',joined[joined.candidate]),('core_audit',joined[joined.core]),('small_cluster_overlap',overlap)]:
        df.to_csv(out/f'{name}.csv',index=False)
    summary=dict(input_dims=len(input_indices()),candidate_n=len(candidates),core_n=len(core),reference_n=71-len(candidates),
        candidates=sorted(candidates),core=sorted(core),small_by_seed={s:sorted(small[s]) for s in seeds},
        max_abs_weight=float(params.max_abs_weight.max()),val_replay_max_error=float(params.val_replay_abs_error.max()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(summary,indent=2))
    (out/'REPORT.md').write_text(f'# {len(input_indices())}维候选家庭检查\n\n小簇并集{len(candidates)}户、核心{len(core)}户：'+ '、'.join(sorted(core))+f'。\n\n213个保存策略均用完整36维重新执行训练/验证回放，验证收益最大误差{params.val_replay_abs_error.max():.3g} EUR/h。数据指标重新从打包的71户CSV计算。\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
