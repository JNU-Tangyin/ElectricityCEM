"""Nonnegative diagonal metric fitted directly to scaled GPS distances.

Five city-grouped folds, city-pair-balanced errors, and whole-city coordinate
permutations. Identical recorded coordinates are excluded from fitting distance
labels, but all households remain in the panel. Same-city distances are audited.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from gps_triplet_pilot import geographic_distances
from policy_replay import tail

from gps_panel import load_panel, result_root, input_indices


def objective(weights, design, target, mass, center, penalty):
    distance = np.sqrt(design @ weights + 1e-12)
    residual = distance - target
    loss = mass @ residual**2 + penalty*np.mean((weights-center)**2)
    gradient = design.T @ (mass*residual/distance)
    gradient += 2*penalty*(weights-center)/len(weights)
    return loss, gradient


def main(args):
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    ids, seeds, raw, *_ = load_panel()
    records = []
    paths = []
    for h in ids:
        path = Path(tail.DATA_DIR)/f"{h}.csv"
        loc = pd.read_csv(path,usecols=["lat","lon"]).drop_duplicates()
        if len(loc)!=1 or not np.isfinite(loc.to_numpy()).all():
            raise ValueError(f"Invalid coordinates for {h}")
        records.append(loc.iloc[0].to_numpy(float))
    points, city = np.unique(records,axis=0,return_inverse=True)
    for seed in seeds:
        files = sorted((Path(tail.paths.SEED_ROOT)/seed).glob("*_expert.npy"))
        if [p.name.removesuffix("_expert.npy") for p in files]!=ids:
            raise ValueError(f"Household IDs do not align: {seed}")
        paths.extend(files)
    pairs = np.triu_indices(len(ids),1)
    counts = np.bincount(city)
    cross = city[pairs[0]] != city[pairs[1]]
    mass_all = 1/(counts[city[pairs[0]]]*counts[city[pairs[1]]])
    rng = np.random.default_rng(args.seed)
    folds = np.array_split(rng.permutation(len(points)),5)
    rows, weight_rows, cluster_rows, same_rows, embeddings = [], [], [], [], {}
    for rep in range(args.permutations+1):
        locations = points if rep==0 else points[rng.permutation(len(points))]
        geo = geographic_distances(locations[city])
        target_all = geo[pairs]
        for fold in [-1,*range(5)]:
            test = np.isin(city,folds[fold]) if fold>=0 else np.zeros(len(ids),bool)
            train = ~test
            fit_mask = cross & train[pairs[0]] & train[pairs[1]]
            mass = mass_all[fit_mask]
            mass /= mass.sum()
            # Equal city-pair median, unaffected by the number of homes per city.
            train_cities = np.unique(city[train])
            city_geo = geographic_distances(locations[train_cities])
            scale = np.median(city_geo[np.triu_indices(len(train_cities),1)])
            y = target_all[fit_mask]/scale
            for seed in seeds:
                scaler = StandardScaler().fit(raw[seed][train])
                z = scaler.transform(raw[seed])
                design = (z[pairs[0]]-z[pairs[1]])**2/z.shape[1]
                base = np.sqrt(design.sum(axis=1))
                coefficient = (mass @ (base[fit_mask]*y))/(mass @ base[fit_mask]**2)
                center = coefficient**2
                fitted = minimize(objective,np.full(z.shape[1],center),
                                  args=(design[fit_mask],y,mass,center,args.penalty),jac=True,
                                  method="L-BFGS-B",bounds=[(1e-10,None)]*z.shape[1],
                                  options={"maxiter":1000,"ftol":1e-12,"gtol":1e-8})
                if not fitted.success:
                    raise RuntimeError(fitted.message)
                predictions = {
                    "constant":np.full(len(base),mass @ y)*scale,
                    "raw_scaled":base*coefficient*scale,
                    "weighted":np.sqrt(design@fitted.x+1e-12)*scale,
                }
                if rep==0:
                    weight_rows.extend((fold,seed,k,v,scale) for k,v in zip(input_indices(),fitted.x))
                for model,pred in predictions.items():
                    masks={"train":fit_mask}
                    if fold>=0:
                        masks["test_to_train"]=cross & (test[pairs[0]] != test[pairs[1]])
                        masks["test_to_test"]=cross & test[pairs[0]] & test[pairs[1]]
                    for split,mask in masks.items():
                        m=mass_all[mask]; m=m/m.sum()
                        error=pred[mask]-target_all[mask]
                        rows.append((rep,fold,seed,model,split,np.sum(mask),
                                     m@np.abs(error),m@error**2))
                    if rep==0 and fold==-1:
                        same_rows.append((seed,model,np.mean(pred[~cross]),int((~cross).sum())))
                if rep==0 and fold==-1:
                    for model,w in [("raw_scaled",np.full(z.shape[1],center)),("weighted",fitted.x)]:
                        embedding=z*np.sqrt(w/z.shape[1])
                        embeddings[f"{seed}_{model}"]=embedding
                        labels=HDBSCAN(min_cluster_size=4,min_samples=2).fit_predict(embedding)
                        cluster_rows.extend((seed,model,h,int(label)) for h,label in zip(ids,labels))
        if rep%10==0:
            print(f"Finished geography {rep}/{args.permutations}",flush=True)
    args.output.mkdir(parents=True)
    results=pd.DataFrame(rows,columns="rep fold seed model split pairs mae_km mse_km2".split())
    results.to_csv(args.output/"fold_metrics.csv",index=False)
    pd.DataFrame(weight_rows,columns="fold seed original_dim weight scale_km".split()).to_csv(args.output/"weights.csv",index=False)
    pd.DataFrame(cluster_rows,columns="seed model source cluster".split()).to_csv(args.output/"labels.csv",index=False)
    pd.DataFrame(same_rows,columns="seed model same_coordinate_mean_predicted_km pairs".split()).to_csv(args.output/"same_coordinate_audit.csv",index=False)
    np.savez_compressed(args.output/"embeddings.npz",households=np.asarray(ids),**embeddings)
    cv=results[results.fold>=0].groupby(["rep","model","split"])[["mae_km","mse_km2"]].mean()
    cv["rmse_km"]=np.sqrt(cv.mse_km2)
    cv.to_csv(args.output/"cv_summary.csv")
    comparisons=[]
    for split in ["test_to_train","test_to_test"]:
        pivot=cv.xs(split,level="split").mae_km.unstack("model")
        improvement=pivot.raw_scaled-pivot.weighted
        null=improvement.iloc[1:]
        comparisons.append((split,improvement.loc[0],null.mean(),null.quantile(.95),
                            (1+(null>=improvement.loc[0]).sum())/(1+len(null))))
    pd.DataFrame(comparisons,columns="split mae_improvement_km null_mean null_q95 p_upper".split()).to_csv(args.output/"permutation_comparison.csv",index=False)
    metadata={"seed":args.seed,"penalty":args.penalty,"permutations":args.permutations,
              "folds":[x.tolist() for x in folds],"households":ids,"city_group":city.tolist(),
              "coordinate_hash":hashlib.sha256(np.asarray(records).tobytes()).hexdigest(),
              "script_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "theta_hashes":{str(p.relative_to(tail.paths.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output/"run_manifest.json").write_text(json.dumps(metadata,indent=2))
    print(cv.loc[0].round(3).to_string())
    print(comparisons)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations",type=int,default=99)
    parser.add_argument("--penalty",type=float,default=.1)
    parser.add_argument("--seed",type=int,default=36)
    parser.add_argument("--output",type=Path,default=result_root()/"exp36_gps_distance_regression")
    args=parser.parse_args()
    if args.permutations<1 or args.penalty<0:
        parser.error("Need positive permutation count and nonnegative penalty")
    with threadpool_limits(limits=1):
        main(args)
