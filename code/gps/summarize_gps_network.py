"""Summarize exp34 with cities equally weighted, and whole-map permutation references."""
from pathlib import Path
import numpy as np
import pandas as pd
from policy_replay import tail

from gps_panel import load_panel, result_root, input_indices


def summarize():
    out=result_root()/"exp34_gps_network_discovery"
    accuracy=pd.read_csv(out/"accuracy.csv")
    grouped=accuracy.groupby(["permutation","model","seed","split","city"]).accuracy.mean()
    cells=grouped.groupby(level=[0,1,2,3]).mean().unstack()
    clusters=pd.read_csv(out/"clusters.csv").set_index(["permutation","model","seed"])
    cells=cells.join(clusters[["coverage","gap_kw","k"]])
    cells.loc[0].to_csv(out/"observed_by_seed.csv")
    avg=cells.groupby(level=[0,1]).mean().where(cells.groupby(level=[0,1]).count()==3)
    pairs=pd.read_csv(out/"stability.csv").groupby(["permutation","model"]).jaccard
    avg["jaccard"]=pairs.mean().where(pairs.count()==3)
    summary=[]
    for model in avg.index.get_level_values("model").unique():
        d=avg.xs(model,level="model")
        for metric in d:
            obs=d.loc[0,metric]
            null=d.loc[d.index!=0,metric].dropna()
            p=(1+(null>=obs).sum())/(len(null)+1) if len(null) and np.isfinite(obs) else np.nan
            summary.append((model,metric,obs,len(null),null.mean(),null.quantile(.95),p))
    result=pd.DataFrame(summary,columns="model metric observed valid_null null_mean null_q95 p_upper".split())
    result.to_csv(out/"summary.csv",index=False)
    print(result.round(4).to_string(index=False))


if __name__=="__main__":
    summarize()
