"""Audit existing Exp34/36 embeddings and labels, without changing either method.

Exp34 embeddings are deterministically reconstructed from its original full-panel
training recipe and checked against saved labels. Exp36 embeddings are loaded.
PCA is display only; raw-label triple intersections are saved without matching.
Run: eenv/bin/python 02.code/multiple_country/gps/plot_gps_distance_clusters.py
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/energy-mpl-cache")
from itertools import combinations
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from gps_network_discovery import network, make_triplets
from gps_triplet_pilot import geographic_distances, fit_triplets
from policy_replay import tail

from gps_panel import load_panel, result_root, input_indices


def align_to_reference(labels, seeds):
    """One-to-one positive-overlap matching; unmatched groups get new colors.

    Different extra groups in different seeds are never assumed equivalent.
    All overlap tables retain original labels, so matching is not needed to audit.
    """
    ref = labels["cem_seed42"]
    classes = sorted(set(ref) - {-1})
    refmap = {int(c): j + 1 for j, c in enumerate(classes)}
    mappings = {"cem_seed42": refmap}
    next_color = len(classes) + 1
    for seed in seeds:
        if seed == "cem_seed42":
            continue
        original = sorted(set(labels[seed]) - {-1})
        mapping = {}
        if original and classes:
            counts = np.array([[np.sum((labels[seed] == c) & (ref == r)) for r in classes] for c in original])
            ii, jj = linear_sum_assignment(-counts)
            mapping = {int(original[i]): refmap[int(classes[j])] for i, j in zip(ii, jj) if counts[i, j] > 0}
        for c in original:
            if int(c) not in mapping:
                mapping[int(c)] = next_color
                next_color += 1
        mappings[seed] = mapping
    aligned = {s: np.array([mappings[s].get(int(v), 0) for v in labels[s]]) for s in seeds}
    return aligned, mappings, next_color - 1


def audit_and_plot(title, embeddings, labels, ids, seeds, coords, out):
    out.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        rebuilt = HDBSCAN(min_cluster_size=4, min_samples=2).fit_predict(embeddings[s])
        assert np.array_equal(rebuilt, labels[s]), (title, s, "Saved-label mismatch")
    aligned, mappings, ncolors = align_to_reference(labels, seeds)
    rain = np.isclose(coords[:, 0], 48.683331) & np.isclose(coords[:, 1], 10.9)
    matrix = np.column_stack([aligned[s] for s in seeds])
    colors = ["#c8ccd2", *[matplotlib.colors.to_hex(plt.get_cmap("tab10")(i)) for i in range(ncolors)]]
    names = ["Unassigned", *[f"C{i}" for i in range(1, ncolors + 1)]]
    legend = [Patch(color=c, label=n) for c, n in zip(colors, names)]
    table = pd.DataFrame({"household": ids, "lat": coords[:, 0], "lon": coords[:, 1], "is_rain": rain})
    for s in seeds:
        table[s + "_original"] = labels[s]
        table[s + "_aligned"] = aligned[s]
    table.to_csv(out / "household_membership.csv", index=False)
    np.savez_compressed(out / "embeddings.npz", households=np.asarray(ids), **embeddings)
    # All actual intersections of the three partitions, independent of color alignment.
    triples = table.groupby([s + "_original" for s in seeds], sort=True)
    cells = []
    for signature, group in triples:
        cells.append({**dict(zip(seeds, map(int, signature))), "n": len(group),
                      "all_assigned": all(v >= 0 for v in signature),
                      "rain_count": int(group.is_rain.sum()), "households": ";".join(group.household)})
    pd.DataFrame(cells).to_csv(out / "all_three_partition_intersections.csv", index=False)
    summary = []
    for c in range(1, ncolors + 1):
        intersection = np.all(matrix == c, axis=1)
        union = np.any(matrix == c, axis=1)
        summary.append({"aligned_cluster": f"C{c}", **{s: int(sum(aligned[s] == c)) for s in seeds},
                        "intersection_three": int(sum(intersection)), "union_three": int(sum(union)),
                        "jaccard_three": sum(intersection) / sum(union),
                        "rain_in_intersection": int(sum(intersection & rain)),
                        "core_households": ";".join(np.array(ids)[intersection])})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "matched_cluster_overlap.csv", index=False)
    projections, pca_info = {}, {}
    for s in seeds:
        pca = PCA(n_components=2)
        projections[s] = pca.fit_transform(embeddings[s])
        pca_info[s] = pca.explained_variance_ratio_.tolist()
    pd.DataFrame([{"seed": s, "household": h, "pc1": xy[0], "pc2": xy[1]}
                  for s in seeds for h, xy in zip(ids, projections[s])]).to_csv(out / "pca_coordinates.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    draw_scatter(axes, seeds, projections, pca_info, aligned, colors, rain)
    fig.suptitle(f"{title} | all 71 households", fontsize=14)
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), bbox_to_anchor=(.5, .045))
    fig.text(.5, .015, "Independent PCA axes; clustering uses the full representation. Colors matched to seed42; black rings: Rain.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.savefig(out / "pca_three_seeds.png", dpi=180)
    plt.close(fig)
    order = sorted(range(len(ids)), key=lambda i: (matrix[i, 2] == 0, matrix[i, 2], matrix[i, 0], matrix[i, 1], ids[i]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 12))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5, ncolors + 1.5), len(colors))
    for ax, indexes in zip(axes, np.array_split(order, 2)):
        m = matrix[indexes]
        ax.imshow(m, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(3), [s.replace("cem_", "") for s in seeds])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(indexes)), [ids[i] + ("  [Rain]" if rain[i] else "") for i in indexes], fontsize=9)
        for row, col in np.ndindex(m.shape):
            ax.text(col, row, "-" if m[row, col] == 0 else f"C{m[row, col]}", ha="center", va="center", fontsize=8,
                    color="black" if m[row, col] in [0, 2] else "white")
        ax.set_yticks(np.arange(-.5, len(indexes)), minor=True)
        ax.grid(which="minor", axis="y", color="white", linewidth=.6)
        ax.tick_params(which="minor", left=False)
    fig.suptitle(title + " | household membership", fontsize=14)
    fig.legend(handles=legend, loc="lower center", ncol=len(legend))
    fig.tight_layout(rect=(0, .035, 1, .96), w_pad=4)
    fig.savefig(out / "household_membership_three_seeds.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    pair_rows = []
    ii, jj = np.triu_indices(len(ids), 1)
    for ax, (a, b) in zip(axes, combinations(seeds, 2)):
        ca = (labels[a][ii] >= 0) & (labels[a][ii] == labels[a][jj])
        cb = (labels[b][ii] >= 0) & (labels[b][ii] == labels[b][jj])
        intersection, union = int(sum(ca & cb)), int(sum(ca | cb))
        score = intersection / union if union else float("nan")
        pair_rows.append({"seed_a": a, "seed_b": b, "cocluster_intersection": intersection,
                          "cocluster_union": union, "cocluster_pair_jaccard": score})
        rows = sorted(set(aligned[a]) - {0}) + [0]
        cols = sorted(set(aligned[b]) - {0}) + [0]
        cross = np.array([[sum((aligned[a] == r) & (aligned[b] == c)) for c in cols] for r in rows])
        ax.imshow(cross, cmap="Blues", vmin=0, vmax=len(ids), aspect="auto")
        for i, j in np.ndindex(cross.shape):
            ax.text(j, i, str(cross[i, j]), ha="center", va="center", color="white" if cross[i, j] > len(ids) / 2 else "black")
        ax.set_xticks(range(len(cols)), [names[c] if c else "None" for c in cols])
        ax.set_yticks(range(len(rows)), [names[c] if c else "None" for c in rows])
        ax.set(xlabel=b.replace("cem_", ""), ylabel=a.replace("cem_", ""), title=f"Co-cluster pair Jaccard: {score:.3f}")
        pd.DataFrame(cross, index=[names[c] for c in rows], columns=[names[c] for c in cols]).to_csv(out / f"overlap_{a}_{b}.csv")
    fig.suptitle(title + " | household overlap counts")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out / "pairwise_overlap_counts.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(pair_rows).to_csv(out / "pairwise_agreement.csv", index=False)
    audit = {"title": title, "all_saved_labels_reproduced_exactly": True,
             "households": len(ids), "rain_households": int(sum(rain)), "reference": "cem_seed42",
             "label_maps": mappings, "pca_variance_ratios": pca_info,
             "counts_by_seed": {s: {"k": len(set(labels[s]) - {-1}), "unassigned": int(sum(labels[s] == -1))} for s in seeds},
             "same_matched_cluster_all_three": int(sum((matrix[:, 0] > 0) & np.all(matrix == matrix[:, :1], axis=1))),
             "unassigned_all_three": int(sum(np.all(matrix == 0, axis=1))),
             "mean_pair_jaccard": float(np.mean([r["cocluster_pair_jaccard"] for r in pair_rows]))}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "README.md").write_text(f"""# {title} 已有结果可视化

保留全部71户，包括Rain18户。原始簇标签逐项验证通过；没有重训CEM或更改表示/聚类参数。
Exp34按原脚本重建全样本表示；Exp36直接读取已保存表示。
HDBSCAN仍在完整表示空间中运行，min_cluster_size=4，min_samples=2。
PCA二维仅供显示，各面板坐标独立；颜色只在同一个实验分支内对齐，不跨分支对应。
以seed42为基准按最大交集一对一匹配；没有正交集的簇、额外簇使用独立颜色。
灰色和0表示未归簇，不是额外簇。全未归簇不等于证明不存在任何结构。
all_three_partition_intersections.csv保存原始标签的所有三重交集，不依赖颜色匹配。
配对Jaccard统计共同归簇的家庭对，排除未归簇点配对；大簇可能主导该指标。
这些结果描述固定设置下的输出，不证明自然离散类型存在；不能仅凭k一致或高整体Jaccard下结论。
""")
    print(title, flush=True)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    return {"title": title, "projections": projections, "pca_info": pca_info, "aligned": aligned, "colors": colors, "rain": rain, "audit": audit}


def draw_scatter(axes, seeds, projections, pca_info, aligned, colors, rain):
    for ax, s in zip(axes, seeds):
        xy = projections[s]
        for c in sorted(set(aligned[s])):
            mask = aligned[s] == c
            ax.scatter(*xy[mask].T, c=colors[c], s=38, alpha=.8, edgecolors="white", linewidths=.4)
        ax.scatter(*xy[rain].T, facecolors="none", edgecolors="black", s=85, linewidths=.7)
        k = len(set(aligned[s]) - {0})
        ax.set(title=f"{s.replace('cem_', '')}: {k} clusters, {sum(aligned[s] == 0)} unassigned",
               xlabel=f"PC1 ({pca_info[s][0]:.1%})", ylabel=f"PC2 ({pca_info[s][1]:.1%})")
        ax.grid(alpha=.15)


def main():
    ids, seeds, raw, *_ = load_panel()
    coords = np.array([pd.read_csv(Path(tail.DATA_DIR) / f"{h}.csv", usecols=["lat", "lon"])
                       [["lat", "lon"]].iloc[0].to_numpy() for h in ids])
    places, city = np.unique(coords, axis=0, return_inverse=True)
    geo = geographic_distances(places[city])
    # The original Exp34 full-panel fold is -1, with triplet RNG seed 343 + fold.
    trips = make_triplets(city, geo, 342)
    e34 = {"diagonal": {}, "network": {}}
    for s in seeds:
        x = StandardScaler().fit_transform(raw[s])
        e34["diagonal"][s] = x * np.sqrt(fit_triplets(x, trips))
        e34["network"][s] = network(x, np.arange(len(ids)), trips)
    base34 = result_root() / "exp34_gps_network_discovery"
    base36 = result_root() / "exp36_gps_distance_regression"
    saved34 = pd.read_csv(base34 / "labels.csv", dtype={"source": str})
    saved36 = pd.read_csv(base36 / "labels.csv", dtype={"source": str})
    saved_emb = np.load(base36 / "embeddings.npz")
    assert saved_emb["households"].tolist() == ids
    e36 = {s: saved_emb[f"{s}_weighted"] for s in seeds}
    specs = [("Exp34 | distance-ranking diagonal", e34["diagonal"], saved34, "diagonal", base34 / "cluster_visualization" / "diagonal"),
             ("Exp34 | distance-ranking network", e34["network"], saved34, "network", base34 / "cluster_visualization" / "network"),
             ("Exp36 | distance-regression diagonal", e36, saved36, "weighted", base36 / "cluster_visualization" / "weighted")]
    results = []
    for title, embeddings, saved, model, out in specs:
        labels = {s: saved[(saved.model == model) & (saved.seed == s)].set_index("source").loc[ids, "cluster"].to_numpy() for s in seeds}
        results.append(audit_and_plot(title, embeddings, labels, ids, seeds, coords, out))
    # Did distance supervision replace, or mostly retain, the existing small tail?
    baseline_comparison = []
    for s in seeds:
        small = {}
        for model in ["raw_scaled", "weighted"]:
            frame = saved36[(saved36.seed == s) & (saved36.model == model)]
            smallest = frame.loc[frame.cluster >= 0, "cluster"].value_counts().idxmin()
            small[model] = set(frame.loc[frame.cluster == smallest, "source"])
        a, b = small["raw_scaled"], small["weighted"]
        baseline_comparison.append({"seed": s, "raw_small_n": len(a), "weighted_small_n": len(b),
                                    "intersection": len(a & b), "union": len(a | b),
                                    "jaccard": len(a & b) / len(a | b)})
    pd.DataFrame(baseline_comparison).to_csv(base36 / "cluster_visualization" / "weighted" / "small_cluster_vs_raw.csv", index=False)
    overview = result_root() / "gps_cluster_visualization_comparison"
    overview.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5))
    for row, result in enumerate(results):
        draw_scatter(axes[row], seeds, result["projections"], result["pca_info"], result["aligned"], result["colors"], result["rain"])
        axes[row, 1].text(.5, 1.16, result["title"], transform=axes[row, 1].transAxes, ha="center", fontsize=13, fontweight="bold")
    fig.suptitle("Existing Exp34 / Exp36 results | same 71 households, three CEM seeds", fontsize=15)
    fig.text(.5, .012, "PCA display only; full-space HDBSCAN unchanged. Gray: unassigned. Rings: Rain. Colors match within each row only.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .95), h_pad=4)
    fig.savefig(overview / "exp34_exp36_pca_overview.png", dpi=180)
    plt.close(fig)
    (overview / "audit_summary.json").write_text(json.dumps([r["audit"] for r in results], indent=2) + "\n")
    print("Overview:", overview)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
