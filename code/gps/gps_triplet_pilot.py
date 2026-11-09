"""Leave-coordinate-out pilot: city-distance triplets supervise a diagonal theta metric."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from policy_replay import build_obs_matrix, env, replay, tail

from gps_panel import load_panel, result_root, input_indices


def geographic_distances(coords):
    lat, lon = np.radians(coords).T
    a = (np.sin((lat[:, None]-lat)/2)**2 + np.cos(lat[:, None])*np.cos(lat)
         * np.sin((lon[:, None]-lon)/2)**2)
    return 12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fit_metric(z, geo, city, rng):
    # Sample cities uniformly, then one household per city (including anchor city).
    groups = [np.flatnonzero(city == c) for c in np.unique(city)]
    triples = []
    for k, anchors in enumerate(groups):
        others = np.delete(np.arange(len(groups)), k)
        for _ in range(32):
            u,v = rng.choice(others, size=2, replace=False)
            a,b,c = rng.choice(anchors), rng.choice(groups[u]), rng.choice(groups[v])
            if np.isclose(geo[a,b],geo[a,c]):
                continue
            if geo[a,b] > geo[a,c]:
                b,c = c,b
            triples.append((a,b,c))
    return fit_triplets(z, triples)


def fit_triplets(z, triples):
    """Fit the same regularized nonnegative diagonal metric for any triplet source."""
    a,b,c = np.asarray(triples).T
    delta = ((z[a]-z[b])**2 - (z[a]-z[c])**2)/np.sqrt(z.shape[1])
    def objective(w):
        loss = 1 + delta @ w
        active = np.maximum(loss,0)
        return (np.mean(active**2) + np.mean((w-1)**2),
                2 * (delta.T @ active/len(active) + (w-1)/len(w)))
    fit = minimize(objective,np.ones(z.shape[1]),jac=True,method="L-BFGS-B",
                   bounds=[(0,None)]*z.shape[1])
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x


def main():
    ids,seeds,raw,*_ = load_panel()
    coords, assays = [], []
    for h in ids:
        df = pd.read_csv(Path(tail.DATA_DIR)/f"{h}.csv",index_col=0,parse_dates=True)
        assert len(df[["lat","lon"]].drop_duplicates()) == 1
        coords.append(df[["lat","lon"]].iloc[0].to_numpy(float))
        e = env.EnergyStorageEnv(df,sell_price_ratio=tail.SELL_RATIO,meta_path=str(tail.paths.META_PATH))
        assays.append(build_obs_matrix(df.iloc[-72:],e.meta,e.all_states,e.local_scaling))
    _,city,counts = np.unique(coords,axis=0,return_inverse=True,return_counts=True)
    geo = geographic_distances(coords)
    rows = []
    for seed in seeds:
        theta = np.stack([np.load(Path(tail.paths.SEED_ROOT)/seed/f"{h}_expert.npy") for h in ids])
        behavior = squareform(np.mean([pdist(replay(theta,x),"cityblock")/72 for x in assays],axis=0))
        for protocol in ["all","exclude_largest"]:
            pool = np.arange(len(ids)) if protocol=="all" else np.flatnonzero(city != counts.argmax())
            for held in np.unique(city[pool]):
                train = pool[city[pool]!=held]
                scaler = StandardScaler().fit(raw[seed][train])
                z = scaler.transform(raw[seed])
                weights = fit_metric(z[train],geo[np.ix_(train,train)],city[train],
                                     np.random.default_rng(970+int(held)))
                i,j = np.triu_indices(len(train),1)
                for a in pool[city[pool]==held]:
                    diff = (z[train]-z[a])**2
                    ranks = geo[a,train[i]]-geo[a,train[j]]
                    valid = abs(ranks)>1e-7
                    for label,w in [("theta",np.ones(len(weights))),("gps_triplet",weights)]:
                        dist = diff @ w
                        order = np.sign(dist[i]-dist[j])*np.sign(ranks)
                        accuracy = np.mean((1+order[valid])/2)
                        neighbors = train[np.argsort(dist,kind="stable")[:5]]
                        rows.append((protocol,seed,held,ids[a],label,accuracy,behavior[a,neighbors].mean()))
    out = result_root()/"exp32_gps_triplet"
    out.mkdir(exist_ok=True)
    detail = pd.DataFrame(rows,columns=["protocol","seed","coordinate_group","household","metric",
                                        "geographic_order_accuracy","neighbor_action_gap_kw"])
    detail.to_csv(out/"heldout.csv",index=False)
    grouped = detail.groupby(["protocol","seed","metric","coordinate_group"])[
        ["geographic_order_accuracy","neighbor_action_gap_kw"]].mean()
    summary = grouped.groupby(level=[0,1,2]).mean()
    summary.to_csv(out/"summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
