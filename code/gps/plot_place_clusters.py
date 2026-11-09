"""Visualize all full36 Exp35 branches without assuming four clusters."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from gps_panel import load_panel, result_root, input_indices
from policy_replay import tail
from gps_place_networks import supcon, place_triplets
from gps_network_discovery import network
from plot_gps_distance_clusters import audit_and_plot


def main():
    ids,seeds,raw,*_=load_panel()
    coords=np.array([pd.read_csv(Path(tail.DATA_DIR)/f'{h}.csv',usecols=['lat','lon'])[['lat','lon']].iloc[0].to_numpy() for h in ids])
    _,city=np.unique(coords,axis=0,return_inverse=True)
    base=result_root()/'exp35_place_networks'
    saved=pd.read_csv(base/'labels.csv',dtype={'source':str})
    trips=place_triplets(city)
    for model in ['raw','supcon','triplet']:
        embeddings={}
        for s in seeds:
            x=StandardScaler().fit_transform(raw[s])
            embeddings[s]=x if model=='raw' else supcon(x,city) if model=='supcon' else network(x,np.arange(len(ids)),trips)
        labels={s:saved[(saved.protocol=='all')&(saved.model==model)&(saved.seed==s)].set_index('source').loc[ids,'cluster'].to_numpy() for s in seeds}
        audit_and_plot(f'Exp35 | {len(input_indices())}D input | {model}',embeddings,labels,ids,seeds,coords,base/'cluster_visualization'/model)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
