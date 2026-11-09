"""Exploratory GPS-supervised clustering of an existing household panel (not held-out prediction).

Compare raw theta, same-location triplets, and between-location distance triplets.
The mixed arm replaces half the same-location constraints with distance constraints.
Every supervised arm gets the same triplet budget and loss. Repeat with the same
household-to-location shuffle across seeds; never split a household's three vectors.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from gps_triplet_pilot import fit_triplets, geographic_distances
from policy_replay import tail

from gps_panel import load_panel, result_root, input_indices


def sample_constraints(city, geo, budget, seed):
    groups = [np.flatnonzero(city == c) for c in np.unique(city)]
    eligible = [k for k, g in enumerate(groups) if len(g) >= 2]
    if not eligible or len(groups) < 3:
        raise ValueError("Need repeated locations and at least three distinct locations.")
    rng = np.random.default_rng(seed)
    by_id, by_distance = [], []
    locations = np.arange(len(groups))
    for i in range(budget):
        k = eligible[i % len(eligible)]
        a, b = rng.choice(groups[k], 2, replace=False)
        c = rng.choice(groups[rng.choice(locations[locations != k])])
        by_id.append((a, b, c))
        for _ in range(100):
            k, u, v = rng.choice(locations, 3, replace=False)
            a, b, c = (rng.choice(groups[j]) for j in (k, u, v))
            if not np.isclose(geo[a,b], geo[a,c]):
                by_distance.append((a,b,c) if geo[a,b] < geo[a,c] else (a,c,b))
                break
        else:
            raise ValueError("No untied distance triplets found.")
    return {"place_id": by_id, "distance": by_distance,
            "mixed": by_id[:budget//2] + by_distance[:budget//2]}


def cocluster(labels, pairs):
    a,b = labels[pairs[0]], labels[pairs[1]]
    return (a >= 0) & (a == b)


def agreements(x, y, pairs):
    a,b = cocluster(x, pairs), cocluster(y, pairs)
    union = (a | b).sum()
    # Every noise point is a singleton, so shared noise is never a cluster.
    singleton = lambda lab: np.where(lab >= 0, lab, -1-np.arange(len(lab)))
    ari = adjusted_rand_score(singleton(x), singleton(y)) if union else np.nan
    return ari, (a & b).sum()/union if union else np.nan


def main(args):
    ids, seeds, raw, *_ = load_panel()
    for seed in seeds:
        names = sorted(p.name.removesuffix("_expert.npy") for p in (Path(tail.paths.SEED_ROOT)/seed).glob("*_expert.npy"))
        if names != ids:
            raise ValueError(f"Household alignment differs for {seed}")
    coords = []
    for h in ids:
        loc = pd.read_csv(Path(tail.DATA_DIR)/f"{h}.csv",usecols=["lat","lon"]).drop_duplicates()
        if len(loc) != 1 or loc.isna().any().any():
            raise ValueError(f"Invalid household coordinates: {h}")
        coords.append(loc[["lat","lon"]].iloc[0].to_numpy(float))
    _, city, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    geo = geographic_distances(coords)
    fields = ["load_mean","pv_mean","zero_load_frac"]
    features = pd.read_csv(tail.paths.FEATURES_CSV, dtype={"household_id":str}).set_index("household_id").loc[ids,fields]
    if not np.isfinite(features).all().all():
        raise ValueError("Nonfinite external profiles.")
    labels_out, fits, agreement, profiles, audit = [], [], [], [], []
    for protocol, idx in [("all", np.arange(len(ids))), ("exclude_largest", np.flatnonzero(city != counts.argmax()))]:
        _, sizes = np.unique(city[idx],return_counts=True)
        audit.append((protocol,len(idx),len(sizes),sum(sizes>1),sum(sizes*(sizes-1)//2)))
        pairs = np.triu_indices(len(idx),1)
        external = {f:pdist(features.iloc[idx][[f]], "cityblock") for f in fields}
        z = {s:StandardScaler().fit_transform(raw[s][idx]) for s in seeds}
        rng = np.random.default_rng(args.seed)
        for rep in range(args.permutations+1):
            # Replicate 0 is the observed geography. A permutation moves whole locations,
            # including their coordinates, between household identities.
            mapped = idx if rep == 0 else rng.permutation(idx)
            constraints = sample_constraints(city[mapped],geo[np.ix_(mapped,mapped)],args.triplets,args.seed)
            arms = ["raw",*constraints] if rep == 0 else list(constraints)
            for arm in arms:
                labs = {}
                for s in seeds:
                    w = np.ones(z[s].shape[1]) if arm=="raw" else fit_triplets(z[s], constraints[arm])
                    lab = HDBSCAN(min_cluster_size=4,min_samples=2).fit_predict(z[s]*np.sqrt(w))
                    labs[s] = lab
                    fits.append((protocol,rep,arm,s,len(set(lab)-{-1}),np.mean(lab>=0)))
                    inside = cocluster(lab,pairs)
                    outside = (lab[pairs[0]] >= 0) & (lab[pairs[1]] >= 0) & (lab[pairs[0]] != lab[pairs[1]])
                    for field,d in external.items():
                        gap = d[outside].mean()-d[inside].mean() if inside.any() and outside.any() else np.nan
                        profiles.append((protocol,rep,arm,s,field,inside.sum(),outside.sum(),gap))
                    if rep == 0:
                        labels_out.extend((protocol,arm,s,ids[i],label) for i,label in zip(idx,lab))
                for a,b in combinations(seeds,2):
                    agreement.append((protocol,rep,arm,a,b,*agreements(labs[a],labs[b],pairs)))

    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    tables = [
        ("sample_support",audit,"protocol n_households n_locations repeated_locations same_location_pairs"),
        ("observed_labels",labels_out,"protocol arm seed household label"),
        ("cluster_coverage",fits,"protocol permutation arm seed n_clusters clustered_fraction"),
        ("seed_agreement",agreement,"protocol permutation arm seed_a seed_b ari co_pair_jaccard"),
        ("profile_separation",profiles,"protocol permutation arm seed field n_within n_between between_minus_within"),
    ]
    for filename,rows,columns in tables:
        pd.DataFrame(rows,columns=columns.split()).to_csv(out/f"{filename}.csv",index=False)
    scores = pd.DataFrame(agreement,columns=tables[3][2].split())
    comparisons = []
    for (protocol, arm), group in scores.groupby(["protocol","arm"]):
        means = group.groupby("permutation").co_pair_jaccard.agg(["mean","count"])
        # Undefined seed-pair agreement is not evidence of stability.
        valid = means["mean"].where(means["count"] == len(list(combinations(seeds,2))))
        null = valid.iloc[1:].dropna()
        comparisons.append((protocol,arm,valid.loc[0],null.mean(),null.quantile(.95),len(null),
                            args.seed,args.triplets,args.permutations))
    pd.DataFrame(comparisons,columns="protocol arm observed_jaccard null_mean null_q95 valid_null seed triplets requested_permutations".split()).to_csv(out/"comparison.csv",index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations",type=int,default=99)
    parser.add_argument("--triplets",type=int,default=1024)
    parser.add_argument("--seed",type=int,default=33)
    parser.add_argument("--output",type=Path,default=result_root()/"exp33_gps_cluster_comparison")
    args = parser.parse_args()
    if args.triplets < 2 or args.triplets % 2 or args.permutations < 0:
        parser.error("triplets must be positive and even; permutations must be nonnegative")
    with threadpool_limits(limits=1):
        main(args)
