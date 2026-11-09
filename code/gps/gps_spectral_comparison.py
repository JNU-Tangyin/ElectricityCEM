"""Fixed-representation spectral audit: raw theta vs existing Exp36 embedding.

Same historical grid: K={2,3,4}, n_neighbors={5,10,15}, kmeans seed=42.
All 71 households and all three CEM seeds; no representation retraining.
Run from project root: eenv/bin/python 02.code/multiple_country/gps/gps_spectral_comparison.py
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/energy-mpl-cache")
import argparse
import hashlib
import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components, laplacian
import sklearn
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from policy_replay import tail

K_VALUES = [2, 3, 4]
NEIGHBORS = [5, 10, 15]
MODELS = ["raw_theta", "exp36_weighted"]

from gps_panel import load_panel, result_root, input_indices


def compare(a, b):
    """Label-invariant whole-partition measures and best membership agreement."""
    ii, jj = np.triu_indices(len(a), 1)
    ca, cb = a[ii] == a[jj], b[ii] == b[jj]
    intersection, union = int(sum(ca & cb)), int(sum(ca | cb))
    ac, bc = np.unique(a), np.unique(b)
    overlap = np.array([[sum((a == x) & (b == y)) for y in bc] for x in ac])
    rows, cols = linear_sum_assignment(-overlap)
    return {"ari": adjusted_rand_score(a, b), "pair_jaccard": intersection / union if union else np.nan,
            "matched_fraction": overlap[rows, cols].sum() / len(a)}


def matched_labels(labels, seeds):
    reference = labels["cem_seed42"]
    ref_classes = sorted(np.unique(reference), key=lambda c: (-sum(reference == c), c))
    result = {}
    for s in seeds:
        classes = np.unique(labels[s])
        overlap = np.array([[sum((labels[s] == a) & (reference == b)) for b in ref_classes] for a in classes])
        rows, cols = linear_sum_assignment(-overlap)
        mapping = {int(classes[i]): int(j) for i, j in zip(rows, cols)}
        result[s] = np.array([mapping[int(v)] for v in labels[s]])
    return result


def main(output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    ids, seeds, raw, *_ = load_panel()
    base = result_root() / "exp36_gps_distance_regression"
    source = base / "embeddings.npz"
    saved = np.load(source)
    assert saved["households"].tolist() == ids and len(ids) == 71
    x = {s: StandardScaler().fit_transform(raw[s]) for s in seeds}
    # Exp36's raw_scaled is a positive global rescaling of the same standardized theta.
    for s in seeds:
        baseline = saved[f"{s}_raw_scaled"]
        coefficient = np.sum(baseline * x[s]) / np.sum(x[s] ** 2)
        assert coefficient > 0 and np.allclose(baseline, x[s] * coefficient)
    embeddings = {"raw_theta": x, "exp36_weighted": {s: saved[f"{s}_weighted"] for s in seeds}}
    coords = np.array([pd.read_csv(Path(tail.DATA_DIR) / f"{h}.csv", usecols=["lat", "lon"])
                       [["lat", "lon"]].iloc[0].to_numpy() for h in ids])
    rain = np.isclose(coords[:, 0], 48.683331) & np.isclose(coords[:, 1], 10.9)
    hdb = pd.read_csv(base / "labels.csv", dtype={"source": str})
    labels, aligned, graphs = {}, {}, {}
    label_rows, cluster_rows, graph_rows, eigen_rows, warning_rows = [], [], [], [], []
    for model in MODELS:
        for q in NEIGHBORS:
            for k in K_VALUES:
                group = {}
                for s in seeds:
                    estimator = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=q,
                                                   eigen_solver="arpack", random_state=42, assign_labels="kmeans", n_init=10)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        lab = estimator.fit_predict(embeddings[model][s])
                    assert len(np.unique(lab)) == k and len(lab) == len(ids)
                    group[s] = lab
                    labels[(model, q, k, s)] = lab
                    for warning in caught:
                        warning_rows.append({"model": model, "n_neighbors": q, "k": k, "seed": s, "warning": str(warning.message)})
                    sizes = np.bincount(lab)
                    cluster_rows.append({"model": model, "n_neighbors": q, "k": k, "seed": s,
                                         "sizes_descending": "/".join(map(str, sorted(sizes, reverse=True))),
                                         "smallest_n": int(min(sizes)), "largest_n": int(max(sizes))})
                    affinity = estimator.affinity_matrix_.toarray()
                    if k == K_VALUES[0]:
                        components, _ = connected_components(affinity, directed=False)
                        eigenvalues = np.linalg.eigvalsh(laplacian(affinity, normed=True))
                        graph_rows.append({"model": model, "n_neighbors": q, "seed": s,
                                           "components": components, "undirected_edges": int(np.sum(np.triu(affinity > 0, 1))),
                                           "lambda2": eigenvalues[1], "lambda3": eigenvalues[2]})
                        eigen_rows.extend({"model": model, "n_neighbors": q, "seed": s, "index": j + 1,
                                           "eigenvalue": v} for j, v in enumerate(eigenvalues))
                        graphs[(model, q, s)] = affinity
                    else:
                        assert np.array_equal(affinity, graphs[(model, q, s)])
                matched = matched_labels(group, seeds)
                for s in seeds:
                    aligned[(model, q, k, s)] = matched[s]
                    label_rows.extend({"model": model, "n_neighbors": q, "k": k, "seed": s,
                                       "household": h, "cluster_original": int(c), "cluster_aligned": int(a) + 1,
                                       "is_rain": bool(r)} for h, c, a, r in zip(ids, group[s], matched[s], rain))
    cross_rows, core_rows, spatial_rows, neighbor_rows, hdb_rows = [], [], [], [], []
    for model in MODELS:
        for q in NEIGHBORS:
            for k in K_VALUES:
                matrix = np.column_stack([aligned[(model, q, k, s)] for s in seeds])
                for a, b in combinations(seeds, 2):
                    cross_rows.append({"model": model, "n_neighbors": q, "k": k, "seed_a": a, "seed_b": b,
                                       **compare(labels[(model, q, k, a)], labels[(model, q, k, b)])})
                for c in range(k):
                    masks = matrix == c
                    intersection, union = np.all(masks, axis=1), np.any(masks, axis=1)
                    core_rows.append({"model": model, "n_neighbors": q, "k": k, "aligned_cluster": c + 1,
                                      **{s: int(sum(masks[:, j])) for j, s in enumerate(seeds)},
                                      "intersection_three": int(sum(intersection)), "union_three": int(sum(union)),
                                      "jaccard_three": sum(intersection) / sum(union),
                                      "rain_in_core": int(sum(intersection & rain)),
                                      "core_households": ";".join(np.array(ids)[intersection])})
                if k == 2:
                    for s in seeds:
                        a = labels[(model, q, k, s)]
                        counts = np.bincount(a)
                        small = a == np.argmin(counts)
                        hm = "raw_scaled" if model == "raw_theta" else "weighted"
                        b = hdb[(hdb.model == hm) & (hdb.seed == s)].set_index("source").loc[ids, "cluster"].to_numpy()
                        options = np.unique(b[b >= 0])
                        hsmall = b == min(options, key=lambda c: sum(b == c)) if len(options) >= 2 else np.zeros(len(b), dtype=bool)
                        hdb_rows.append({"model": model, "n_neighbors": q, "seed": s,
                                         "spectral_small_n": int(sum(small)), "hdbscan_small_n": int(sum(hsmall)),
                                         "intersection": int(sum(small & hsmall)), "union": int(sum(small | hsmall)),
                                         "jaccard": sum(small & hsmall) / sum(small | hsmall)})
        for k in K_VALUES:
            for s in seeds:
                for qa, qb in combinations(NEIGHBORS, 2):
                    neighbor_rows.append({"model": model, "k": k, "seed": s, "neighbors_a": qa, "neighbors_b": qb,
                                          **compare(labels[(model, qa, k, s)], labels[(model, qb, k, s)])})
    for q in NEIGHBORS:
        for k in K_VALUES:
            for s in seeds:
                spatial_rows.append({"n_neighbors": q, "k": k, "seed": s,
                                     **compare(labels[(MODELS[0], q, k, s)], labels[(MODELS[1], q, k, s)])})
    output.mkdir(parents=True)
    tables = {"labels": label_rows, "cluster_sizes": cluster_rows, "graph_diagnostics": graph_rows,
              "graph_eigenvalues": eigen_rows, "cross_seed_agreement": cross_rows, "cluster_core": core_rows,
              "raw_vs_exp36": spatial_rows, "neighbor_sensitivity": neighbor_rows, "k2_vs_hdbscan": hdb_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(warning_rows, columns=["model", "n_neighbors", "k", "seed", "warning"]).to_csv(output / "warnings.csv", index=False)
    cross = pd.DataFrame(cross_rows)
    summary = cross.groupby(["model", "n_neighbors", "k"])[["ari", "pair_jaccard", "matched_fraction"]].mean().reset_index()
    summary.to_csv(output / "cross_seed_summary.csv", index=False)
    make_plots(output, seeds, ids, rain, embeddings, aligned, summary, pd.DataFrame(core_rows), pd.DataFrame(spatial_rows))
    manifest = {"households": ids, "cem_seeds": seeds, "k_values": K_VALUES, "n_neighbors": NEIGHBORS,
                "spectral_random_state": 42, "assign_labels": "kmeans", "n_init": 10, "eigen_solver": "arpack",
                "affinity": "nearest_neighbors", "sklearn_version": sklearn.__version__, "run_count": len(cluster_rows),
                "source_embeddings": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "retrained_representation": False, "retrained_cem": False,
                "raw_scaled_verified_positive_global_rescaling": True}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text("""# 原始θ与Exp36：固定表示的谱聚类对照

71户，保留Rain18户。3个CEM seed。原始θ取当前配置的参数维度并在各seed内标准化；Exp36直接读已有加权表示，不再逐维标准化（否则会抵消学得权重）。
沿用历史Python谱聚类网格：K=2/3/4，n_neighbors=5/10/15；2空间×3CEM seed×9设置=54次拟合。K是指定值，不是自动发现的类别数。
assign_labels=kmeans，固定random_state=42和n_init=10；不重训CEM或表示。此次没有扫描谱聚类自身的随机种子。
近邻图使用sklearn的nearest_neighbors口径（包含自身后再对称化）；图连通分量和全部特征值均保存。
cluster_core.csv给出以seed42为基准、最大成员交集匹配后的三次共同成员。C1为seed42最大簇。颜色只在一个设置内部可比较。
cross_seed_agreement.csv与raw_vs_exp36.csv的ARI/家庭对Jaccard不依赖簇编号；整体分数可能被大簇主导。
neighbor_sensitivity.csv检查构图参数变化；k2_vs_hdbscan.csv比较两个聚类器的小簇成员。
labels.csv保留全部54次输出，未按得分挑选结果。图中K=2、邻居10的详细展示由运行前固定，不是最佳设置。
PCA二维仅展示已保存的输入空间，谱聚类在完整输入维度表示上构图；图上重叠不代表高维没有分离。
本轮是描述性方法对照，未做重采样、连续无簇零参照、GPS置换或独立行为验证，不能据此确认自然类别或宣布最优方法。
""")
    print(summary.round(4).to_string(index=False))
    print("\nK=2 matched cluster cores:")
    print(pd.DataFrame(core_rows).query("k == 2").drop(columns="core_households").to_string(index=False))
    print("\nGraph connected components:", pd.DataFrame(graph_rows).components.value_counts().to_dict())
    print("\nMean raw/weighted agreement:\n", pd.DataFrame(spatial_rows).groupby("k")[["ari", "pair_jaccard"]].mean())
    print("Saved:", output)


def make_plots(out, seeds, ids, rain, embeddings, labels, summary, cores, spatial):
    colors = ["#287bb5", "#ec8b22", "#37975b", "#955eb0"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for row, model in enumerate(MODELS):
        for col, (metric, title, vmin, vmax) in enumerate([
                ("ari", "Cross-seed mean ARI", -.1, 1),
                ("pair_jaccard", "Cross-seed pair Jaccard", 0, 1),
                ("minimum_core", "Smallest cluster-wise 3-seed Jaccard", 0, 1)]):
            frame = (cores.groupby(["model", "n_neighbors", "k"]).jaccard_three.min().reset_index(name=metric)
                     if metric == "minimum_core" else summary)
            grid = frame[frame.model == model].pivot(index="n_neighbors", columns="k", values=metric).reindex(index=NEIGHBORS, columns=K_VALUES).to_numpy()
            ax = axes[row, col]
            ax.imshow(grid, cmap="Blues", vmin=vmin, vmax=vmax)
            for i, j in np.ndindex(grid.shape):
                ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", color="white" if grid[i,j] > .6 else "black")
            ax.set_xticks(range(3), K_VALUES)
            ax.set_yticks(range(3), NEIGHBORS)
            ax.set(xlabel="Specified K", ylabel=f"{model}\nn_neighbors", title=title)
    fig.suptitle("All fixed spectral settings | 71 households, three CEM seeds", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / "all_settings_stability.png", dpi=180)
    plt.close(fig)
    # Fixed middle neighborhood, K=2; not selected using the observed scores.
    q, k = 10, 2
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for row, model in enumerate(MODELS):
        for col, s in enumerate(seeds):
            pca = PCA(n_components=2)
            xy = pca.fit_transform(embeddings[model][s])
            lab = labels[(model, q, k, s)]
            ax = axes[row, col]
            for c in range(k):
                ax.scatter(*xy[lab == c].T, c=colors[c], s=35, alpha=.8, edgecolors="white", linewidths=.4)
            ax.scatter(*xy[rain].T, facecolors="none", edgecolors="black", s=85, linewidths=.7)
            ax.set(title=f"{model} | {s.replace('cem_', '')} | sizes {sum(lab == 0)}/{sum(lab == 1)}",
                   xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax.grid(alpha=.15)
    fig.suptitle("Spectral clustering: specified K=2, n_neighbors=10 | PCA display only", fontsize=14)
    fig.text(.5, .015, "Colors aligned within each row to seed42. Rings: Rain. PCA axes independent; no representation retraining.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(out / "k2_neighbors10_pca.png", dpi=180)
    plt.close(fig)
    # All 71 IDs and all three CEM seeds in each space.
    fig, axes = plt.subplots(1, 2, figsize=(12, 17))
    for ax, model in zip(axes, MODELS):
        matrix = np.column_stack([labels[(model, q, k, s)] for s in seeds])
        order = sorted(range(len(ids)), key=lambda i: (matrix[i, 2], matrix[i, 0], matrix[i, 1], ids[i]))
        matrix = matrix[order]
        ax.imshow(matrix, cmap=ListedColormap(colors[:2]), vmin=-.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(3), [s.replace("cem_", "") for s in seeds])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(ids)), [ids[i] + (" [Rain]" if rain[i] else "") for i in order], fontsize=7)
        ax.set_title(model, pad=28)
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, f"C{matrix[i,j]+1}", ha="center", va="center", fontsize=7, color="white" if matrix[i,j] == 0 else "black")
    fig.suptitle("Same household across seeds | specified K=2, n_neighbors=10", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97), w_pad=4)
    fig.savefig(out / "k2_neighbors10_membership.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=result_root() / "exp36_gps_distance_regression" / "spectral_comparison")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.output)
