"""Test GPS ordering on held-out cities and explore full-panel behavioral clusters."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from gps_cluster_comparison import agreements, cocluster
from gps_triplet_pilot import geographic_distances, fit_triplets
from policy_replay import tail, env, build_obs_matrix, replay

from gps_panel import load_panel, result_root, input_indices


def forward(x, layers):
    states = [x]
    for i,(w,b) in enumerate(layers):
        x = x @ w + b
        states.append(np.maximum(x,0) if i < len(layers)-1 else x)
        x = states[-1]
    return states


def network(z, train, triples):
    # Four shared dense layers, Euclidean triplet margin 1, SGD 0.01, 10 epochs.
    rng = np.random.default_rng(34)
    layers = [(rng.normal(0,np.sqrt(2/d),(d,64)), np.zeros(64)) for d in (z.shape[1],64,64,64)]
    for _ in range(10):
        for batch in np.array_split(rng.permutation(len(triples)),32):
            states = forward(z[train], layers)
            out = states[-1]
            a,b,c = triples[batch].T
            ab, ac = out[a]-out[b], out[a]-out[c]
            near,far = np.linalg.norm(ab,axis=1),np.linalg.norm(ac,axis=1)
            active = (1+near-far>0)[:,None]/len(batch)
            pos = ab/np.maximum(near[:,None],1e-10)*active
            neg = ac/np.maximum(far[:,None],1e-10)*active
            g = np.zeros_like(out)
            np.add.at(g,a,pos-neg); np.add.at(g,b,-pos); np.add.at(g,c,neg)
            for i in reversed(range(len(layers))):
                w,bias = layers[i]
                if i < len(layers)-1:
                    g *= states[i+1]>0
                dw,db = states[i].T@g, g.sum(axis=0)
                g = g@w.T
                w -= .01*dw
                bias -= .01*db
    return forward(z,layers)[-1]


def make_triplets(city, geo, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(city==v) for v in np.unique(city)]
    triples = []
    while len(triples)<1024:
        a,b,c = (rng.choice(groups[k]) for k in rng.choice(len(groups),3,replace=False))
        if np.isclose(geo[a,b],geo[a,c]):
            continue
        triples.append((a,b,c) if geo[a,b]<geo[a,c] else (a,c,b))
    return np.asarray(triples)


def order_scores(d, geo, city, anchors, train):
    # Equal weight per candidate-city pair and (in the summary) per anchor city.
    x,y = np.triu_indices(len(train),1)
    i,j=train[x],train[y]
    count=np.bincount(city[train])
    weights=1/(count[city[i]]*count[city[j]])
    rows=[]
    for a in anchors:
        gd=geo[a,i]-geo[a,j]
        valid=(city[a]!=city[i])&(city[a]!=city[j])&(abs(gd)>1e-7)
        correct=(1+np.sign(gd)*np.sign(d[a,i]-d[a,j]))/2
        rows.append(np.average(correct[valid],weights=weights[valid]))
    return rows


def main():
    ids,seeds,raw,*_=load_panel()
    coords, assays=[],[]
    for h in ids:
        df=pd.read_csv(Path(tail.DATA_DIR)/f"{h}.csv", index_col=0,parse_dates=True)
        pts=df[["lat","lon"]].drop_duplicates()
        assert pts.shape==(1,2) and np.isfinite(pts).all().all()
        coords.append(pts.iloc[0].to_numpy())
        e=env.EnergyStorageEnv(df,sell_price_ratio=tail.SELL_RATIO,meta_path=str(tail.paths.META_PATH))
        assays.append(build_obs_matrix(df.iloc[-72:],e.meta,e.all_states,e.local_scaling))
    places,city=np.unique(coords,axis=0,return_inverse=True)
    rng=np.random.default_rng(34)
    folds=np.array_split(rng.permutation(len(places)),5)
    behavior={}
    for s in seeds:
        files=sorted((Path(tail.paths.SEED_ROOT)/s).glob("*_expert.npy"))
        assert [f.name.removesuffix("_expert.npy") for f in files]==ids
        th=np.stack([np.load(f) for f in files])
        behavior[s]=np.mean([pdist(replay(th,x),"cityblock")/72 for x in assays],axis=0)
    panel=np.arange(len(ids)); pairs=np.triu_indices(len(ids),1)
    scoring=[]; clusters=[]; stability=[]; memberships=[]
    for rep in range(40):
        mapping=places if rep==0 else places[rng.permutation(len(places))]
        geo=geographic_distances(mapping[city])
        for fold in range(-1,5):
            test=panel[np.isin(city,folds[fold])] if fold>=0 else panel[:0]
            train=np.setdiff1d(panel,test)
            trip=make_triplets(city[train],geo[np.ix_(train,train)],343+fold)
            labels={}
            for s in seeds:
                z=StandardScaler().fit(raw[s][train]).transform(raw[s])
                models={"diagonal":z*np.sqrt(fit_triplets(z[train],trip)),
                        "network":network(z,train,trip)}
                if rep==0:
                    models["raw"]=z
                for model,embedding in models.items():
                    if fold>=0:
                        d=squareform(pdist(embedding))
                        for split,anchors in [("train",train),("test",test)]:
                            acc=order_scores(d,geo,city,anchors,train)
                            scoring.extend((rep,fold,s,model,split,city[a],ids[a],v) for a,v in zip(anchors,acc))
                    else:
                        lab=HDBSCAN(min_cluster_size=4,min_samples=2).fit_predict(embedding)
                        labels.setdefault(model,[]).append(lab)
                        inside=cocluster(lab,pairs)
                        outside=(lab[pairs[0]]>=0)&(lab[pairs[1]]>=0)&(lab[pairs[0]]!=lab[pairs[1]])
                        within=behavior[s][inside].mean() if inside.any() else np.nan
                        between=behavior[s][outside].mean() if outside.any() else np.nan
                        clusters.append((rep,s,model,len(set(lab)-{-1}),np.mean(lab>=0),within,between,between-within))
                        if rep==0:
                            memberships.extend((s,model,h,l) for h,l in zip(ids,lab))
            for model,labs in labels.items():
                for i,j in combinations(range(len(seeds)),2):
                    stability.append((rep,model,seeds[i],seeds[j],*agreements(labs[i],labs[j],pairs)))
        if rep%5==0: print(f"Completed geography {rep}/39",flush=True)
    out=result_root()/"exp34_gps_network_discovery"
    out.mkdir(exist_ok=True)
    for name,rows,cols in [
        ("accuracy",scoring,"permutation fold seed model split city source accuracy"),
        ("clusters",clusters,"permutation seed model k coverage within_kw between_kw gap_kw"),
        ("stability",stability,"permutation model seed_a seed_b ari jaccard"),
        ("labels",memberships,"seed model source cluster")]:
        pd.DataFrame(rows,columns=cols.split()).to_csv(out/f"{name}.csv",index=False)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
