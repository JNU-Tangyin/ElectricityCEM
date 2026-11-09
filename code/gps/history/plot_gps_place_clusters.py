"""Reproduce Exp35 all/SupCon embeddings; visualize saved labels and household overlap.

Run from project root: eenv/bin/python 02.code/multiple_country/gps/plot_gps_place_clusters.py
PCA is display only. HDBSCAN runs on the original eight-dimensional embedding.
Colors are matched post hoc to seed42 by maximum household overlap, excluding noise.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/energy-mpl-cache")
from itertools import combinations
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from gps_place_networks import supcon, tail


def main():
    ids, seeds, raw, *_ = tail.load_panel()
    base = Path(tail.paths.SHAPE) / "exp35_place_networks"
    out = base / "cluster_visualization"
    out.mkdir(exist_ok=True)
    saved = pd.read_csv(base / "labels.csv", dtype={"source": str})
    saved = saved[(saved.protocol == "all") & (saved.model == "supcon")]
    coords = np.array([pd.read_csv(Path(tail.DATA_DIR) / f"{h}.csv", usecols=["lat", "lon"])
                       [["lat", "lon"]].iloc[0].to_numpy() for h in ids])
    locations, city, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    rain = np.isclose(coords[:, 0], 48.683331) & np.isclose(coords[:, 1], 10.9)
    embeddings, labels = {}, {}
    for s in seeds:
        embeddings[s] = supcon(StandardScaler().fit_transform(raw[s]), city)
        labels[s] = saved[saved.seed == s].set_index("source").loc[ids, "cluster"].to_numpy()
        rebuilt = HDBSCAN(min_cluster_size=4, min_samples=2).fit_predict(embeddings[s])
        assert np.array_equal(labels[s], rebuilt), f"Saved labels differ: {s}"
    reference = "cem_seed42"
    ref = labels[reference]
    classes = sorted(set(ref) - {-1})
    assert len(classes) == 4
    aligned, maps = {}, {}
    for s in seeds:
        original = sorted(set(labels[s]) - {-1})
        assert len(original) == len(classes)
        overlap = np.array([[np.sum((labels[s] == a) & (ref == b)) for b in classes] for a in original])
        ii, jj = linear_sum_assignment(-overlap)
        mapping = {int(original[i]): int(classes[j]) + 1 for i, j in zip(ii, jj)}
        maps[s] = mapping
        aligned[s] = np.array([mapping.get(int(v), 0) for v in labels[s]])
    matrix = np.column_stack([aligned[s] for s in seeds])
    table = pd.DataFrame({"household": ids, "lat": coords[:, 0], "lon": coords[:, 1], "is_rain": rain})
    for s in seeds:
        table[s + "_original"] = labels[s]
        table[s + "_aligned"] = aligned[s]
        for j in range(8):
            table[f"{s}_embedding_{j}"] = embeddings[s][:, j]
    table["same_cluster_all_three"] = (matrix[:, 0] > 0) & np.all(matrix == matrix[:, :1], axis=1)
    table.to_csv(out / "household_membership_and_embeddings.csv", index=False)
    rows = []
    for c in range(1, 5):
        core = np.all(matrix == c, axis=1)
        union = np.any(matrix == c, axis=1)
        rows.append({"cluster": f"C{c}", **{s: int(sum(aligned[s] == c)) for s in seeds},
                     "intersection_three": int(sum(core)), "union_three": int(sum(union)),
                     "jaccard_three": sum(core) / sum(union), "rain_in_core": int(sum(core & rain)),
                     "core_households": ";".join(np.array(ids)[core])})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "cluster_overlap_summary.csv", index=False)
    colors = ["#c8ccd2", "#2778b4", "#e58b22", "#32965a", "#9457b2"]
    names = ["Unassigned", "C1", "C2", "C3", "C4"]
    legend = [Patch(color=c, label=n) for c, n in zip(colors, names)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    projections = []
    for ax, s in zip(axes, seeds):
        pca = PCA(n_components=2)
        xy = pca.fit_transform(embeddings[s])
        for c in range(5):
            mask = aligned[s] == c
            ax.scatter(*xy[mask].T, c=colors[c], s=38, alpha=.8, edgecolors="white", linewidths=.4)
        ax.scatter(*xy[rain].T, facecolors="none", edgecolors="black", s=85, linewidths=.7)
        ax.set(title=f"{s.replace('cem_', '')}: 4 clusters, {sum(aligned[s] == 0)} unassigned",
               xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
               ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.grid(alpha=.15)
        projections.extend({"seed": s, "household": h, "pc1": x, "pc2": y,
                            "pc1_variance": pca.explained_variance_ratio_[0],
                            "pc2_variance": pca.explained_variance_ratio_[1]} for h, (x, y) in zip(ids, xy))
    fig.suptitle("Exp35 | GPS place-supervised representation | all 71 households", fontsize=14)
    fig.legend(handles=legend, loc="lower center", ncol=5, bbox_to_anchor=(.5, .045))
    fig.text(.5, .015, "Independent PCA axes per seed; colors matched to seed42 by household overlap. Black rings: Rain. Clustering uses 8D.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94))
    fig.savefig(out / "pca_three_seeds.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(projections).to_csv(out / "pca_coordinates.csv", index=False)
    order = sorted(range(len(ids)), key=lambda i: (matrix[i, 2] == 0, matrix[i, 2], matrix[i, 0], matrix[i, 1], ids[i]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 12))
    for ax, indexes in zip(axes, np.array_split(order, 2)):
        m = matrix[indexes]
        ax.imshow(m, cmap=ListedColormap(colors), vmin=-.5, vmax=4.5, aspect="auto")
        ax.set_xticks(range(3), [s.replace("cem_", "") for s in seeds])
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(indexes)), [ids[i] + ("  [Rain]" if rain[i] else "") for i in indexes], fontsize=9)
        for row in range(len(indexes)):
            for col in range(3):
                ax.text(col, row, "-" if m[row, col] == 0 else f"C{m[row, col]}", ha="center", va="center", fontsize=8,
                        color="black" if m[row, col] in [0, 2] else "white")
        ax.set_yticks(np.arange(-.5, len(indexes), 1), minor=True)
        ax.grid(which="minor", axis="y", color="white", linewidth=.6)
        ax.tick_params(which="minor", left=False)
    fig.suptitle("Same household, three seeds: do the cluster assignments agree?", fontsize=14)
    fig.legend(handles=legend, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, .035, 1, .96), w_pad=4)
    fig.savefig(out / "household_membership_three_seeds.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    pair_rows = []
    display = [1, 2, 3, 4, 0]
    pairs = np.triu_indices(len(ids), 1)
    for ax, (a, b) in zip(axes, combinations(seeds, 2)):
        cross = np.array([[sum((aligned[a] == i) & (aligned[b] == j)) for j in display] for i in display])
        ax.imshow(cross, cmap="Blues", vmin=0, vmax=max(counts))
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cross[i, j]), ha="center", va="center", color="white" if cross[i, j] > 10 else "black")
        ax.set_xticks(range(5), ["C1", "C2", "C3", "C4", "None"])
        ax.set_yticks(range(5), ["C1", "C2", "C3", "C4", "None"])
        ax.set(xlabel=b.replace("cem_", ""), ylabel=a.replace("cem_", ""))
        ca = (aligned[a][pairs[0]] > 0) & (aligned[a][pairs[0]] == aligned[a][pairs[1]])
        cb = (aligned[b][pairs[0]] > 0) & (aligned[b][pairs[0]] == aligned[b][pairs[1]])
        jaccard = sum(ca & cb) / sum(ca | cb)
        pair_rows.append({"seed_a": a, "seed_b": b, "cocluster_pair_jaccard": jaccard})
        ax.set_title(f"Co-cluster pair Jaccard: {jaccard:.3f}")
        pd.DataFrame(cross, index=[names[c] for c in display], columns=[names[c] for c in display]).to_csv(out / f"overlap_{a}_{b}.csv")
    fig.suptitle("Household counts between seeds (None = unassigned, not a cluster)")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out / "pairwise_overlap_counts.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(pair_rows).to_csv(out / "pairwise_agreement.csv", index=False)
    facts = {"saved_labels_reproduced_exactly": True, "reference": reference, "label_maps": maps,
             "rain_households": int(sum(rain)), "assigned_all_three": int(sum(np.all(matrix > 0, axis=1))),
             "same_cluster_all_three": int(sum(table.same_cluster_all_three)),
             "unassigned_all_three": int(sum(np.all(matrix == 0, axis=1))),
             "mapper_random_seed": 35, "cem_seeds": seeds}
    (out / "audit.json").write_text(json.dumps(facts, indent=2) + "\n")
    (out / "README.md").write_text("""# Exp35 全部71户、同地点 SupCon 的跨 seed 可视化

- 保留 Rain。这是同地点标签监督的 Exp35，不能代表地理距离监督实验。
- 从原始 θ 重建映射；三个 seed 的 HDBSCAN 标签均与已有 labels.csv 逐项完全相同。没有重新训练 CEM。
- HDBSCAN 在 8 维空间运行，参数 min_cluster_size=4、min_samples=2。PCA 仅用于显示，三个面板坐标轴独立，不能按绝对位置比较。
- 簇颜色以 seed42 为基准，通过最大成员交集的一对一匹配对齐。匹配是事后的；编号本身无语义。灰色/0 表示未归簇，不能视为第五簇。
- cluster_overlap_summary.csv 提供三次交集、并集、Jaccard 和共同成员 ID。家庭配对 Jaccard 不依赖簇编号。
- 家庭表包含原始标签、对齐标签及重建的 8 维表示。图中没有对重叠点添加抖动。
- 三个 seed 指 CEM 的 0、1、42；表示网络初始化均固定为35。当前检查未包含映射网络初始化稳定性。
- 这些图和成员交集只能描述已获得划分，不证明存在自然离散类别；同地点监督造成的 Rain 重合不能独立证明跨地点策略类型。
""")
    print(summary.to_string(index=False))
    print(json.dumps(facts, indent=2))
    print(pd.DataFrame(pair_rows).to_string(index=False))
    print(out)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
