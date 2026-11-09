"""Does GPS add predictive information about common-input policy behavior?

Household profiles use the first 191 hours; targets use the last 72 hours.
This is an exploratory panel analysis: those dates were used by the original CEM.
No household or coordinate group is removed. Primary ridge alpha is fixed at 10,
with 1 and 100 reported as sensitivity settings rather than selected on test data.
"""
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from policy_replay import tail, env, build_obs_matrix, replay

METRICS = ["charge_kwh_per_day", "charge_share_10_16", "soc_lower_fraction"]
FEATURES = ["log_load_mean", "log_pv_mean", "zero_load_fraction",
            "load_cv", "cloud_mean", "radiation_mean"]

from gps_panel import load_panel, result_root, input_indices


def city_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1/counts[inverse]
    return weights*len(groups)/weights.sum()


def main():
    out = result_root()/"exp37_gps_behavior_increment"
    if out.exists():
        raise FileExistsError(out)
    ids, seeds, *_ = load_panel()
    frames, features, coords, observations = [], [], [], []
    for h in ids:
        df = pd.read_csv(Path(tail.DATA_DIR)/f"{h}.csv", index_col=0, parse_dates=True)
        assert len(df)==287
        loc = df[["lat","lon"]].drop_duplicates()
        assert len(loc)==1 and np.isfinite(loc.to_numpy()).all()
        coords.append(loc.iloc[0].to_numpy(float))
        profile = df.iloc[:191]
        load = profile.user_load_wh.to_numpy()
        features.append([np.log1p(load.mean()),np.log1p(profile.pv_gen_wh.mean()),
                         np.mean(load==0),load.std()/max(load.mean(),1e-10),
                         profile.cloud_cover.mean(),profile.shortwave_radiation.mean()])
        assert profile.index[-1] < df.index[-72]
        frames.append(df)
        environment = env.EnergyStorageEnv(df,sell_price_ratio=tail.SELL_RATIO,meta_path=str(tail.paths.META_PATH))
        observations.append(build_obs_matrix(df.iloc[-72:],environment.meta,
                                             environment.all_states,environment.local_scaling))
    coords, x = np.asarray(coords), np.asarray(features)
    assert np.isfinite(x).all()
    _, city = np.unique(coords,axis=0,return_inverse=True)
    cities = np.unique(city)
    target_parts, detailed, profiles = [], [], []
    for seed in seeds:
        files = sorted((Path(tail.paths.SEED_ROOT)/seed).glob("*_expert.npy"))
        assert [p.name.removesuffix("_expert.npy") for p in files]==ids
        theta = np.stack([np.load(p) for p in files])
        context_results = []
        for host, df, obs in zip(ids,frames,observations):
            state_trace = []
            def observe(t, soc, intended, actual):
                state_trace.append(soc.copy())
            power = replay(theta,obs,observe=observe)
            charging = np.maximum(power,0)
            daily_charge = charging.sum(axis=1)/3
            hours = df.iloc[-72:].hour_of_day.to_numpy()
            total_charge = charging.sum(axis=1)
            daytime = charging[:,(hours>=10)&(hours<16)].sum(axis=1)
            share = np.divide(daytime,total_charge,out=np.zeros_like(daytime),where=total_charge>1e-10)
            lower = (np.asarray(state_trace)<=env.MIN_SOC+1e-8).mean(axis=0)
            metrics = np.column_stack([daily_charge,share,lower])
            assert np.isfinite(metrics).all()
            context_results.append(metrics)
            for h, values, total in zip(ids,metrics,total_charge):
                detailed.append((seed,host,h,*values,total<=1e-10))
        target = np.mean(context_results,axis=0)
        target_parts.append(target)
        profiles.extend((seed,h,*values) for h,values in zip(ids,target))
    y = np.concatenate(target_parts,axis=1)
    columns = [(s,m) for s in seeds for m in METRICS]

    stability, spatial = [], []
    for k, metric in enumerate(METRICS):
        for a,b in combinations(range(len(seeds)),2):
            r = spearmanr(target_parts[a][:,k],target_parts[b][:,k]).statistic
            stability.append((metric,seeds[a],seeds[b],r))
        for s,target in zip(seeds,target_parts):
            for j,coordinate in enumerate(["latitude","longitude"]):
                spatial.append((s,metric,coordinate,spearmanr(coords[:,j],target[:,k]).statistic))

    predictions, estimates = [], {}
    models = {"profiles": x, "profiles_gps": np.column_stack([x,coords])}
    baseline = np.empty_like(y)
    for alpha in [1.,10.,100.]:
        fitted = {model:np.empty_like(y) for model in models}
        for held in cities:
            test = city==held
            train = ~test
            weights = city_weights(city[train])
            baseline[test] = np.average(y[train],axis=0,weights=weights)
            for model, inputs in models.items():
                scaler = StandardScaler().fit(inputs[train],sample_weight=weights)
                reg = Ridge(alpha=alpha).fit(scaler.transform(inputs[train]),y[train],sample_weight=weights)
                fitted[model][test] = reg.predict(scaler.transform(inputs[test]))
        estimates[alpha] = fitted
        for model, pred in fitted.items():
            for j,(seed,metric) in enumerate(columns):
                predictions.extend((alpha,model,seed,metric,h,int(c),actual,guess,base)
                                   for h,c,actual,guess,base in zip(ids,city,y[:,j],pred[:,j],baseline[:,j]))

    bootstrap = np.random.default_rng(37).integers(len(cities),size=(5000,len(cities)))
    summary = []
    for alpha, fitted in estimates.items():
        for k, metric in enumerate(METRICS):
            cols = np.arange(k,y.shape[1],len(METRICS))
            mse = {}
            for model, pred in {**fitted,"constant":baseline}.items():
                mse[model] = np.array([((pred[city==c][:,cols]-y[city==c][:,cols])**2).mean()
                                       for c in cities])
            delta = mse["profiles"]-mse["profiles_gps"]
            lo,hi = np.quantile(delta[bootstrap].mean(axis=1),[.025,.975])
            summary.append((alpha,metric,np.sqrt(mse["constant"].mean()),
                            np.sqrt(mse["profiles"].mean()),np.sqrt(mse["profiles_gps"].mean()),
                            1-mse["profiles"].mean()/mse["constant"].mean(),
                            1-mse["profiles_gps"].mean()/mse["constant"].mean(),
                            delta.mean(),lo,hi))

    out.mkdir()
    for name,rows,header in [
        ("behavior_by_context",detailed,"seed host source "+" ".join(METRICS)+" no_charge"),
        ("behavior",profiles,"seed source "+" ".join(METRICS)),
        ("stability",stability,"metric seed_a seed_b rho"),
        ("spatial_association",spatial,"seed metric coordinate descriptive_rho"),
        ("predictions",predictions,"alpha model seed metric source city actual predicted constant"),
        ("summary",summary,"alpha metric constant_rmse profiles_rmse gps_rmse profiles_cv_score gps_cv_score mse_gain gain_ci_lo gain_ci_hi")]:
        pd.DataFrame(rows,columns=header.split()).to_csv(out/f"{name}.csv",index=False)
    pd.DataFrame(x,index=pd.Index(ids,name="household"),columns=FEATURES).to_csv(out/"input_profiles.csv")
    metadata={"primary_alpha":10,"sensitivity_alphas":[1,100],"folds":"leave one coordinate group out",
              "n_households":len(ids),"n_locations":len(cities),"baseline":"train-city-balanced mean",
              "ci":"5000 city-cluster bootstraps of fixed out-of-fold errors; conditional, models not refitted",
              "profile_hours":191,"behavior_hours":72,"features":FEATURES,
              "note":"Three seeds averaged inside each city error; same 71 contexts for every policy; no multiple-test significance claim."}
    (out/"protocol.json").write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(summary,columns="alpha metric constant_rmse profiles_rmse gps_rmse profiles_cv_score gps_cv_score mse_gain gain_ci_lo gain_ci_hi".split()).round(5).to_string(index=False))
    print(pd.DataFrame(stability,columns="metric seed_a seed_b rho".split()).round(3).to_string(index=False))


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
