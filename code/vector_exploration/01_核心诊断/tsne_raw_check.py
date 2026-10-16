# -*- coding: utf-8 -*-
"""
直接对 71 户标准化 θ 跑 t-SNE 降到 2 维，看数据本身会不会自己长出团块。
扫多个 perplexity，不只看一次结果；umap_raw_check.py 是同一目的的另一种方法。

读取与画图在 embedding_common.py，两个脚本共用，这里只做 t-SNE 这步降维。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_基础"))
import paths
import argparse

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from embedding_common import (

    build_feature_matrix, load_theta,
    plot_panel_plain,
)

DEFAULT_VECTORS_DIR = paths.CANONICAL
PERPLEXITIES = [5, 15, 30]
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="t-SNE embedding of raw theta vectors.")
    parser.add_argument("--vectors_dir", type=str, default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--output_dir", type=str, default=None, help="Defaults to shape/exp6_embedding/tsne")
    parser.add_argument("--label", type=str, default=None, help="Extra tag for output filename/title, e.g. run identifier")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(paths.SHAPE, "exp6_embedding", "tsne")
    os.makedirs(output_dir, exist_ok=True)

    ids, theta = load_theta(args.vectors_dir)
    x_std = build_feature_matrix(theta)
    print(f">>> Loaded {len(ids)} households, standardized feature dim {x_std.shape[1]}")

    fig, axes = plt.subplots(1, len(PERPLEXITIES), figsize=(7.5 * len(PERPLEXITIES), 6))
    for ax, perp in zip(axes, PERPLEXITIES):
        coords = TSNE(n_components=2, perplexity=perp, random_state=SEED, init="pca").fit_transform(x_std)
        plot_panel_plain(ax, ids, coords)
        ax.set_title(f"perplexity={perp}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")


    title_tag = f" [{args.label}]" if args.label else ""
    fig.suptitle(f"t-SNE embedding, 71 households, {x_std.shape[1]}D standardized theta{title_tag}", fontsize=13)
    fig.tight_layout(rect=None)

    fname = f"tsne_plain{'_' + args.label if args.label else ''}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=130)
    print(f">>> Saved: {out_path}")


if __name__ == "__main__":
    main()
