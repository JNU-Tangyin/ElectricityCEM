"""Same-place supervision: observed and shuffled place IDs, fixed clustering and external profiles."""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from gps_network_discovery import network, forward
from gps_cluster_comparison import agreements, cocluster
from policy_replay import tail, env, build_obs_matrix, replay

from gps_panel import load_panel, result_root, input_indices


def supcon(z, city):
    rng = np.random.default_rng(35)
    layers = [(rng.normal(0, np.sqrt(2/i), (i,j)), np.zeros(j))
              for i,j in [(z.shape[1],32),(32,8)]]
    same = (city[:,None] == city[None,:]) & ~np.eye(len(city),dtype=bool)
    npos = same.sum(axis=1)
    target = same / np.maximum(npos[:,None],1)
    anchor = np.where(npos>0,1/(npos+1),0)
    anchor /= anchor.sum()
    for step in range(201):
        states = forward(z,layers)
        norm = np.maximum(np.linalg.norm(states[-1],axis=1,keepdims=True),1e-12)
        unit = states[-1]/norm
        if step == 200:
            return unit
        logits = (unit @ unit.T)/.1
        np.fill_diagonal(logits,-np.inf)
        residual = anchor[:,None]*(softmax(logits,axis=1)-target)
        g = (residual+residual.T)@unit/.1
        g = (g-unit*(g*unit).sum(axis=1,keepdims=True))/norm
        for i in [1,0]:
            if i==0:
                g *= states[1]>0
            w,b = layers[i]
            dw,db = states[i].T@g,g.sum(axis=0)
            g = g@w.T
            w -= .03*dw
            b -= .03*db


def place_triplets(city):
    rng = np.random.default_rng(35)
    groups = [np.flatnonzero(city==v) for v in np.unique(city)]
    repeated = [k for k,g in enumerate(groups) if len(g)>1]
    trips=[]
    for i in range(1024):
        k=repeated[i%len(repeated)]
        a,b=rng.choice(groups[k],2,replace=False)
        other=rng.integers(len(groups)-1)
        c=rng.choice(groups[other+int(other>=k)])
        trips.append((a,b,c))
    return np.array(trips)


def place_fit(embedding,city):
    d=squareform(pdist(embedding))
    means=[]
    for v in np.unique(city):
        members=np.flatnonzero(city==v)
        if len(members)<2:
            continue
        comparisons=[]
        for a in members:
            neg=d[a,city!=v]
            pos=d[a,members[members!=a]]
            comparisons.append(((1+np.sign(neg[None,:]-pos[:,None]))/2).mean())
        means.append(np.mean(comparisons))
    return np.mean(means)


def main():
    ids,seeds,raw,*_=load_panel()
    coords,obs,profiles=[],[],[]
    for h in ids:
        df=pd.read_csv(Path(tail.DATA_DIR)/f"{h}.csv",index_col=0,parse_dates=True)
        pts=df[["lat","lon"]].drop_duplicates()
        assert pts.shape==(1,2) and np.isfinite(pts).all().all()
        coords.append(pts.iloc[0].to_numpy())
        e=env.EnergyStorageEnv(df,sell_price_ratio=tail.SELL_RATIO,meta_path=str(tail.paths.META_PATH))
        obs.append(build_obs_matrix(df.iloc[-72:],e.meta,e.all_states,e.local_scaling))
        if not profiles:
            timestamps=df.index
        assert df.index.equals(timestamps)
        profile=df.groupby("hour_of_day").user_load_wh.mean().reindex(range(24)).to_numpy()
        assert np.isfinite(profile).all() and profile.sum()>0
        profiles.append(profile/profile.sum())
    _,city,counts=np.unique(coords,axis=0,return_inverse=True,return_counts=True)
    source_theta={}
    for s in seeds:
        files=sorted((Path(tail.paths.SEED_ROOT)/s).glob("*_expert.npy"))
        assert [f.name.removesuffix("_expert.npy") for f in files]==ids
        source_theta[s]=np.stack([np.load(f) for f in files])
    out=result_root()/"exp35_place_networks"
    out.mkdir(exist_ok=True)
    rows,labels,pairrows=[],[],[]
    for protocol,idx in [("all",np.arange(len(ids))),("exclude_largest",np.flatnonzero(city!=counts.argmax()))]:
        pairs=np.triu_indices(len(idx),1)
        action={s:np.mean([pdist(replay(th[idx],obs[i]),"cityblock")/72 for i in idx],axis=0)
                for s,th in source_theta.items()}
        load=pdist(np.asarray(profiles)[idx],"cityblock")/2
        z={s:StandardScaler().fit_transform(raw[s][idx]) for s in seeds}
        for model in ["raw","supcon","triplet"]:
            rng=np.random.default_rng(35)
            for rep in range(40 if model!="raw" else 1):
                loc=city[idx] if rep==0 else rng.permutation(city[idx])
                trips=place_triplets(loc) if model=="triplet" else None
                labs=[]
                for s in seeds:
                    emb=z[s] if model=="raw" else (supcon(z[s],loc) if model=="supcon" else
                                                  network(z[s],np.arange(len(idx)),trips))
                    assert np.isfinite(emb).all()
                    lab=HDBSCAN(min_cluster_size=4,min_samples=2).fit_predict(emb)
                    labs.append(lab)
                    inside=cocluster(lab,pairs)
                    a,b=lab[pairs[0]],lab[pairs[1]]
                    outside=(a>=0)&(b>=0)&(a!=b)
                    def gap(metric):
                        return metric[outside].mean()-metric[inside].mean() if outside.any() and inside.any() else np.nan
                    rows.append((protocol,model,rep,s,place_fit(emb,loc),len(set(lab)-{-1}),
                                 (lab>=0).mean(),gap(action[s]),gap(load)))
                    if rep==0:
                        labels.extend((protocol,model,s,ids[i],l) for i,l in zip(idx,lab))
                for i,j in combinations(range(len(seeds)),2):
                    pairrows.append((protocol,model,rep,seeds[i],seeds[j],agreements(labs[i],labs[j],pairs)[1]))
            print(protocol,model,"complete",flush=True)
    d=pd.DataFrame(rows,columns="protocol model rep seed place_fit k coverage action_gap_kw load_gap".split())
    p=pd.DataFrame(pairrows,columns="protocol model rep seed_a seed_b jaccard".split())
    d.to_csv(out/"cluster_metrics.csv",index=False)
    p.to_csv(out/"agreement.csv",index=False)
    pd.DataFrame(labels,columns="protocol model seed source cluster".split()).to_csv(out/"labels.csv",index=False)
    keys=["protocol","model","rep"]
    g=d.groupby(keys)[["place_fit","coverage","action_gap_kw","load_gap"]]
    avg=g.mean().where(g.count()==3)
    j=p.groupby(keys).jaccard
    avg["jaccard"]=j.mean().where(j.count()==3)
    summary=[]
    for (pr,m),frame in avg.groupby(level=[0,1]):
        for metric in frame:
            v=frame[metric].iloc[0]
            null=frame[metric].iloc[1:].dropna()
            summary.append((pr,m,metric,v,len(null),null.mean(),null.quantile(.95)))
    pd.DataFrame(summary,columns="protocol model metric observed valid_null null_mean null_q95".split()).to_csv(out/"summary.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
