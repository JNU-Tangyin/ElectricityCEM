"""Exp39: coordinate-balanced spherical regions, grouped OOF triplet supervision.
Primary: 35D, K=4, all; sensitivity K=3/5 and exclude_Rain. 199 permutations.
Architecture/loss follow senior's four-layer 64D triplet model. Triplets fixed at
256 per fit, batch32, 10 epochs, SGD .01, margin1. Not a literal URL reproduction.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/energy-gps-mpl')
from pathlib import Path
import argparse, json, hashlib, time
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,f1_score,adjusted_rand_score
from threadpoolctl import threadpool_limits
from gps_panel import load_panel
from gps_context import PACKAGE_ROOT,tail,RESULTS_ROOT


def train_batch(x,triples,epochs=10):
    """Independent networks vectorized over replicates; no shared fitted weights."""
    batches=len(triples);rng=np.random.default_rng(34)
    layers=[(np.broadcast_to(rng.normal(0,np.sqrt(2/d),(d,64)),(batches,d,64)).copy(),np.zeros((batches,1,64))) for d in (x.shape[1],64,64,64)]
    batchidx=np.arange(batches)[:,None]
    for _ in range(epochs):
        for ix in np.array_split(rng.permutation(triples.shape[1]),8):
            states=[np.broadcast_to(x,(batches,*x.shape))]
            for i,(w,b) in enumerate(layers):
                y=states[-1]@w+b;states.append(np.maximum(y,0) if i<3 else y)
            a,p,n=triples[:,ix].transpose(2,0,1)
            ab=states[-1][batchidx,a]-states[-1][batchidx,p];ac=states[-1][batchidx,a]-states[-1][batchidx,n]
            near=np.linalg.norm(ab,axis=2);far=np.linalg.norm(ac,axis=2)
            active=(1+near-far>0)[...,None]/len(ix)
            pos=ab/np.maximum(near[...,None],1e-10)*active;neg=ac/np.maximum(far[...,None],1e-10)*active
            g=np.zeros_like(states[-1]);np.add.at(g,(batchidx,a),pos-neg);np.add.at(g,(batchidx,p),-pos);np.add.at(g,(batchidx,n),neg)
            for i in range(3,-1,-1):
                w,b=layers[i]
                if i<3:g*=states[i+1]>0
                dw=states[i].transpose(0,2,1)@g;db=g.sum(axis=1,keepdims=True);g=g@w.transpose(0,2,1)
                w-=.01*dw;b-=.01*db
    return layers


def transform(x,layers):
    y=np.broadcast_to(x,(len(layers[0][0]),*x.shape))
    for i,(w,b) in enumerate(layers):
        y=y@w+b
        if i<3:y=np.maximum(y,0)
    return y


def sample_triples(city,labels,seed):
    rng=np.random.default_rng(seed);unique=np.unique(city)
    groups={int(c):np.flatnonzero(city==c) for c in unique}
    eligible=[c for c in unique if sum(labels[unique]==labels[c])>=2]
    assert eligible
    rows=[]
    for _ in range(256):
        c=int(rng.choice(eligible));positive=unique[(labels[unique]==labels[c])&(unique!=c)];negative=unique[labels[unique]!=labels[c]]
        p=int(rng.choice(positive));n=int(rng.choice(negative))
        rows.append([rng.choice(groups[c]),rng.choice(groups[p]),rng.choice(groups[n])])
    return np.asarray(rows,dtype=int)


def predict(embedding,train,city,labels,k):
    out=[]
    for b,z in enumerate(embedding):
        centroids=[]
        for r in range(k):
            members=np.unique(city[train][labels[b,city[train]]==r])
            assert len(members)
            centroids.append(np.mean([z[train & (city==c)].mean(axis=0) for c in members],axis=0))
        out.append(np.argmin(((z[:,None]-np.array(centroids)[None])**2).sum(axis=2),axis=1))
    return np.array(out)


def run_case(k,protocol,permutations,out):
    begin=time.monotonic();ids,seeds,raw,*_=load_panel()
    coords=np.array([pd.read_csv(tail.DATA_DIR/f'{h}.csv',usecols=['lat','lon'])[['lat','lon']].iloc[0].to_numpy() for h in ids])
    places,city,counts=np.unique(coords,axis=0,return_inverse=True,return_counts=True)
    lat,lon=np.radians(places).T;xyz=np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
    region=KMeans(k,n_init=50,random_state=0).fit_predict(xyz)
    rain=np.isclose(places[:,0],48.683331)&np.isclose(places[:,1],10.9)
    keep=np.ones(len(ids),bool) if protocol=='all' else ~rain[city]
    pool=np.flatnonzero(keep);active=np.unique(city[pool]);city_local=city[pool]
    assert np.bincount(region[city_local],minlength=k).min()>=8
    # Fixed geographic definition shared by both protocols; Rain removed only from training/evaluation.
    rng=np.random.default_rng(39);label_maps=np.tile(region,(permutations+1,1))
    for b in range(1,permutations+1):label_maps[b,active]=rng.permutation(region[active])
    for labels in label_maps:np.testing.assert_array_equal(np.bincount(labels[active],minlength=k),np.bincount(region[active],minlength=k))
    folds=json.loads((RESULTS_ROOT/'35d/exp36_gps_distance_regression/run_manifest.json').read_text())['folds']
    predictions={s:{m:np.full((permutations+1,len(pool)),-1,dtype=int) for m in ['raw_centroid','triplet_mlp']} for s in seeds}
    triplet_audit=[]
    for fold,held in enumerate(folds):
        test=np.isin(city_local,held);train=~test
        triples=np.stack([sample_triples(city_local[train],labels,3900+fold) for labels in label_maps])
        # Positive coordinates differ from anchors for every observed and shuffled task.
        assert np.all(city_local[train][triples[:,:,0]]!=city_local[train][triples[:,:,1]])
        triplet_audit.append({'fold':fold,'n_train_households':int(train.sum()),'n_test_households':int(test.sum()),'n_train_coordinates':len(np.unique(city_local[train]))})
        for s in seeds:
            x=StandardScaler().fit(raw[s][pool][train]).transform(raw[s][pool])
            base=np.broadcast_to(x,(permutations+1,*x.shape))
            predictions[s]['raw_centroid'][:,test]=predict(base,train,city_local,label_maps,k)[:,test]
            for first in range(0,permutations+1,50):
                last=min(first+50,permutations+1)
                layers=train_batch(x[train],triples[first:last]);z=transform(x,layers)
                assert np.isfinite(z).all()
                predictions[s]['triplet_mlp'][first:last,test]=predict(z,train,city_local,label_maps[first:last],k)[:,test]
        print(f'K={k} {protocol}: fold {fold+1}/5 complete',flush=True)
    scores=[];detail=[]
    weights=1/np.bincount(city_local)[city_local]
    for s in seeds:
        for model,pred in predictions[s].items():
            assert (pred>=0).all()
            for b in range(permutations+1):
                truth=label_maps[b,city_local]
                scores.append(dict(k=k,protocol=protocol,seed=s,model=model,rep=b,
                    balanced_accuracy=balanced_accuracy_score(truth,pred[b],sample_weight=weights),
                    macro_f1=f1_score(truth,pred[b],sample_weight=weights,labels=np.arange(k),average='macro',zero_division=0)))
            detail.extend(dict(k=k,protocol=protocol,seed=s,model=model,household=ids[h],city=int(city[h]),region=int(region[city[h]]),prediction=int(guess)) for h,guess in zip(pool,pred[0]))
    # Full-panel real-label embeddings and clustering are descriptive, separate from OOF scores.
    clusterrows=[];clusterstats=[];embeddings={};trips=sample_triples(city_local,region,3999)[None]
    for s in seeds:
        x=StandardScaler().fit_transform(raw[s][pool]);z=transform(x,train_batch(x,trips))[0];embeddings[s]=z
        for method in ['hdbscan','kmeans']:
            lab=HDBSCAN(min_cluster_size=4,min_samples=2).fit_predict(z) if method=='hdbscan' else KMeans(k,n_init=50,random_state=42).fit_predict(z)
            clusterrows.extend(dict(k=k,protocol=protocol,seed=s,method=method,household=ids[h],region=int(region[city[h]]),cluster=int(l)) for h,l in zip(pool,lab))
            clusterstats.append(dict(k=k,protocol=protocol,seed=s,method=method,n_clusters=len(set(lab)-{-1}),coverage=float((lab>=0).mean())))
    out.mkdir(parents=True)
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False);pd.DataFrame(detail).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(clusterrows).to_csv(out/'cluster_labels.csv',index=False);pd.DataFrame(clusterstats).to_csv(out/'cluster_counts.csv',index=False)
    pd.DataFrame(triplet_audit).to_csv(out/'fold_audit.csv',index=False)
    np.savez_compressed(out/'embeddings.npz',households=np.array(ids)[pool],**embeddings)
    pd.DataFrame({'coordinate':np.arange(len(places)),'lat':places[:,0],'lon':places[:,1],'households':counts,'region':region,'is_rain':rain}).to_csv(out/'regions.csv',index=False)
    np.save(out/'permuted_region_maps.npy',label_maps)
    print(f'Finished K={k} {protocol} in {time.monotonic()-begin:.1f}s',flush=True)


def main(args):
    assert os.environ.get('ENERGY_GPS_INPUT_DIMS')=='35'
    out=RESULTS_ROOT/'region_supervision_exp39'
    target=out/f'k{args.k}_{args.protocol}'
    if target.exists():raise FileExistsError(target)
    run_case(args.k,args.protocol,args.permutations,target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--k',type=int,choices=[3,4,5],required=True);p.add_argument('--protocol',choices=['all','exclude_Rain'],required=True);p.add_argument('--permutations',type=int,default=199)
    with threadpool_limits(limits=1):main(p.parse_args())
