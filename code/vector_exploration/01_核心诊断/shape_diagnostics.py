"""
诊断点云的聚集性、最近邻尺度、Isomap 低维结构和 PCA 方差，并模拟两类备择分布估计
部分检验的功效。功效结论仅适用于模拟覆盖的备择。高维下 Hopkins 使用无幂次形式；
二维嵌入仅用于展示，不能据此确认簇结构。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_基础"))
import paths
import argparse
import glob

import numpy as np

import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import connected_components


SEED_ROOT = paths.SEED_ROOT
OUT_DIR = None  # 输出按文件名经 paths.shape_out() 路由到各实验目录
N_MC = 500
N_POWER = 200
N_NEIGHBORS_LIST = [5, 7, 10, 15, 20]
RV_THRESHOLD = 0.2
RNG = np.random.default_rng(42)


def load_raw_theta(seed_dir):
    files = sorted(glob.glob(os.path.join(SEED_ROOT, seed_dir, "*_expert.npy")))
    ids = [os.path.basename(f).replace("_expert.npy", "") for f in files]
    theta = np.array([np.load(f) for f in files])
    keep = paths.dims_geometry(theta.shape[1])
    return ids, theta[:, keep]


def build_shared_scaler(seed_dirs):
    allX = np.vstack([load_raw_theta(sd)[1] for sd in seed_dirs])
    return StandardScaler().fit(allX)


def load_theta(seed_dir, scaler):
    ids, raw = load_raw_theta(seed_dir)
    return ids, scaler.transform(raw)


def hopkins(X, rng, m_frac=0.1, power=False):
    """Hopkins 统计量。power=True 为经典的 D 次幂形式，在本数据的维度下会饱和。"""
    n, d = X.shape
    m = max(5, int(m_frac * n))
    tree = cKDTree(X)
    U = rng.uniform(X.min(axis=0), X.max(axis=0), size=(m, d))
    u = tree.query(U, k=1)[0]
    idx = rng.choice(n, m, replace=False)
    w = tree.query(X[idx], k=2)[0][:, 1]
    if not power:
        su, sw = u.sum(), w.sum()
        return su / (su + sw) if (su + sw) > 0 else np.nan
    eps = 1e-12
    log_u, log_w = d * np.log(u + eps), d * np.log(w + eps)
    mx = max(log_u.max(), log_w.max())
    su, sw = np.exp(log_u - mx).sum(), np.exp(log_w - mx).sum()
    return su / (su + sw) if (su + sw) > 0 else np.nan


def mean_nn(X):
    """平均最近邻距离。每次蒙特卡洛出一个值构成零分布；多次模拟的逐点距离不能拼成
    一个大样本做秩检验，那些点并不独立。"""
    return cKDTree(X).query(X, k=2)[0][:, 1].mean()


def gaussian_null(X, rng):
    return rng.multivariate_normal(X.mean(axis=0), np.cov(X, rowvar=False), size=len(X))


def uniform_null(X, rng):
    return rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)


def manifold_alternative(X, rng, noise=0.3):
    """非线性另类假设：弯曲二维螺旋 + 一维厚度，嵌入高维后加各向同性噪声，
    并缩放到与真实数据同尺度，以免尺度差异本身造成检出。"""
    n, d = X.shape
    t = rng.uniform(0, 3 * np.pi, size=n)
    M = np.zeros((n, d))
    M[:, 0] = t * np.cos(t)
    M[:, 1] = t * np.sin(t)
    M[:, 2] = rng.uniform(-3, 3, size=n)
    M += rng.normal(scale=noise, size=(n, d))
    return M / M.std() * X.std()


def graph_components(X, n_neighbors):
    """kNN 图的连通分量数。>1 时 Isomap 的测地距离没有良好定义，其残差方差不可用。"""
    G = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity", include_self=False)
    n_comp, _ = connected_components(G, directed=False)
    return n_comp


def isomap_residual(X, n_neighbors, max_dim=12):
    rows = []
    for k in range(1, max_dim + 1):
        iso = Isomap(n_neighbors=n_neighbors, n_components=k)
        emb = iso.fit_transform(X)
        geo = iso.dist_matrix_[np.triu_indices(len(X), k=1)]
        r = np.corrcoef(geo, pdist(emb))[0, 1]
        rows.append({"n_components": k, "residual_variance": 1 - r ** 2})
    return pd.DataFrame(rows)


def d_at_threshold(rv_series, thr=RV_THRESHOLD):
    below = rv_series[rv_series < thr]
    return int(below.index.min()) if len(below) else np.nan


def power_analysis(X, rng):
    n, d = X.shape
    mu, cov = X.mean(axis=0), np.cov(X, rowvar=False)
    null_h = np.array([hopkins(rng.multivariate_normal(mu, cov, size=n), rng) for _ in range(N_MC)])
    null_nn = np.array([mean_nn(rng.multivariate_normal(mu, cov, size=n)) for _ in range(N_MC)])
    thr_h = np.quantile(null_h, 0.95)
    # NN 必须双侧：团块让最近邻距离变长，流形让它变短，单侧准则对其中一类没有功效。
    thr_nn_lo = np.quantile(null_nn, 0.025)
    thr_nn_hi = np.quantile(null_nn, 0.975)

    rows = []
    for frac in [0.10, 0.20, 0.35, 0.50]:
        m = max(2, int(round(frac * n)))
        for delta in [0, 2, 4, 8]:
            h_hit = nn_hit = 0
            for _ in range(N_POWER):
                M = rng.multivariate_normal(mu, cov, size=n)
                v = rng.normal(size=d); v /= np.linalg.norm(v)
                M[:m] += delta * v
                h_hit += hopkins(M, rng) > thr_h
                v_nn = mean_nn(M)
                nn_hit += (v_nn < thr_nn_lo) or (v_nn > thr_nn_hi)
            rows.append({"alternative": "gaussian_blob_shift", "minority_frac": frac,
                         "param": delta, "hopkins_power": h_hit / N_POWER,
                         "nn_power": nn_hit / N_POWER})
    # 流形假设下必须同时测 Isomap：它才是负责低维结构的那个诊断。拟合慢，故重复次数
    # 少于前两者。
    n_iso_rep = max(20, N_POWER // 10)
    for noise in [0.1, 0.3, 0.6]:
        h_hit = nn_hit = 0
        for _ in range(N_POWER):
            M = manifold_alternative(X, rng, noise=noise)
            h_hit += hopkins(M, rng) > thr_h
            v_nn = mean_nn(M)
            nn_hit += (v_nn < thr_nn_lo) or (v_nn > thr_nn_hi)
        iso_hit = 0
        for _ in range(n_iso_rep):
            M = manifold_alternative(X, rng, noise=noise)
            if graph_components(M, 10) > 1:
                continue
            rv = isomap_residual(M, n_neighbors=10, max_dim=6).set_index("n_components")["residual_variance"]
            d02 = d_at_threshold(rv)
            iso_hit += (not np.isnan(d02)) and d02 <= 3
        rows.append({"alternative": "curved_manifold", "minority_frac": np.nan,
                     "param": noise, "hopkins_power": h_hit / N_POWER,
                     "nn_power": nn_hit / N_POWER,
                     "isomap_d02_le3_power": iso_hit / n_iso_rep})
    return pd.DataFrame(rows)


def out_path(args, filename):
    """本脚本一次产出分属 4 个实验的文件，故未给 --output_dir 时按文件名分流。"""
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        return os.path.join(args.output_dir, filename)
    return paths.shape_out(filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=OUT_DIR,
                    help="给定则所有输出写进该目录；默认按文件名分流到 shape/expN_*")
    args = ap.parse_args()

    seed_dirs = sorted(d for d in os.listdir(SEED_ROOT)
                       if os.path.isdir(os.path.join(SEED_ROOT, d))
                       and glob.glob(os.path.join(SEED_ROOT, d, "*_expert.npy")))
    print(f">>> seeds: {seed_dirs}")
    scaler = build_shared_scaler(seed_dirs)
    print(">>> 已在全部 seed 上 fit 共享 scaler（跨 seed 坐标尺度一致）\n")

    # ---------- (0) 功效分析 ----------
    print("=" * 74)
    print(">>> (0) 功效分析（两类另类假设；结论只对模拟过的那一类成立）")
    print("=" * 74)
    _, X0 = load_theta(seed_dirs[0], scaler)
    pw = power_analysis(X0, RNG)
    ga = pw[pw.alternative == "gaussian_blob_shift"]
    print("(a) 高斯云 + 平移子团 —— Hopkins 检出率 (行=少数比例, 列=分离度):")
    print(ga.pivot(index="minority_frac", columns="param", values="hopkins_power").to_string())
    print("\n(a) 同上 —— 最近邻汇总量 检出率:")
    print(ga.pivot(index="minority_frac", columns="param", values="nn_power").to_string())
    print("\n(b) 弯曲流形（螺旋带）:")
    mf = pw[pw.alternative == "curved_manifold"]
    print(mf[["param", "hopkins_power", "nn_power", "isomap_d02_le3_power"]]
          .rename(columns={"param": "noise"}).to_string(index=False))
    pw.to_csv(out_path(args, "power_analysis.csv"), index=False)

    # ---------- (1)(2) ----------
    print("\n" + "=" * 74)
    print(">>> (1) Hopkins   (2) 最近邻汇总量（均以蒙特卡洛汇总量作零分布）")
    print("=" * 74)
    rows = []
    for sd in seed_dirs:
        _, X = load_theta(sd, scaler)
        h_vals = np.array([hopkins(X, RNG) for _ in range(50)])
        h_obs = h_vals.mean()
        h_pow = np.mean([hopkins(X, RNG, power=True) for _ in range(20)])
        h_g = np.array([hopkins(gaussian_null(X, RNG), RNG) for _ in range(N_MC)])
        h_u = np.array([hopkins(uniform_null(X, RNG), RNG) for _ in range(N_MC)])
        nn_obs = mean_nn(X)
        nn_g = np.array([mean_nn(gaussian_null(X, RNG)) for _ in range(N_MC)])
        rows.append({
            "seed": sd, "hopkins_obs": h_obs,
            "hopkins_q025": np.quantile(h_vals, 0.025),
            "hopkins_q975": np.quantile(h_vals, 0.975),
            "hopkins_Dpower_saturated": h_pow,
            "hopkins_unif_mean": h_u.mean(),
            "p_vs_uniform": (1 + (h_u >= h_obs).sum()) / (1 + N_MC),
            "hopkins_gauss_mean": h_g.mean(),
            "hopkins_gaussian_reference_tail": (1 + (h_g >= h_obs).sum()) / (1 + N_MC),
            "nn_obs": nn_obs, "nn_gauss_mean": nn_g.mean(),
            "nn_p_two_sided": 2 * min(
                (1 + (nn_g <= nn_obs).sum()) / (1 + N_MC),
                (1 + (nn_g >= nn_obs).sum()) / (1 + N_MC)),
        })
    df1 = pd.DataFrame(rows)
    print(df1.round(4).to_string(index=False))
    print("\n  p_vs_uniform    : Hopkins 的正式检验（是否偏离空间均匀随机）")
    print("  hopkins_gaussian_reference_tail : 拟合单高斯云后，观察到这么高的 Hopkins 是否罕见；")
    print("                                    这不是 Hopkins 对高斯零假设的 p 值，两者是不同问题")
    df1.to_csv(out_path(args, "hopkins_nn.csv"), index=False)

    # ---------- (3) ----------
    print("\n" + "=" * 74)
    print(">>> (3) Isomap 残差方差（扫多个 n_neighbors）")
    print("=" * 74)
    iso_rows, d02_rows = [], []
    for sd in seed_dirs:
        _, X = load_theta(sd, scaler)
        for k in N_NEIGHBORS_LIST:
            rv = isomap_residual(X, n_neighbors=k).set_index("n_components")["residual_variance"]
            for nc, v in rv.items():
                iso_rows.append({"seed": sd, "n_neighbors": k, "n_components": nc,
                                 "residual_variance": v})
            d02_rows.append({"seed": sd, "n_neighbors": k, "d_0.2": d_at_threshold(rv)})
    df_iso = pd.DataFrame(iso_rows)
    df_d02 = pd.DataFrame(d02_rows)
    print("d_0.2 = 残差方差首次 < 0.2 的维度数（NaN = 12维内未达到）:")
    print(df_d02.pivot(index="n_neighbors", columns="seed", values="d_0.2").to_string())
    print("\n残差方差曲线（n_neighbors=10）:")
    sub = df_iso[df_iso.n_neighbors == 10]
    print(sub.pivot(index="n_components", columns="seed", values="residual_variance").round(3).to_string())
    df_iso.to_csv(out_path(args, "isomap_residual.csv"), index=False)
    df_d02.to_csv(out_path(args, "isomap_d02.csv"), index=False)

    # ---------- (4) ----------
    print("\n" + "=" * 74)
    print(">>> (4) PCA scree")
    print("=" * 74)
    pca_rows = []
    for sd in seed_dirs:
        _, X = load_theta(sd, scaler)
        p = PCA(n_components=12).fit(X)
        for i, (ev, cum) in enumerate(zip(p.explained_variance_ratio_,
                                          np.cumsum(p.explained_variance_ratio_)), start=1):
            pca_rows.append({"seed": sd, "n_components": i, "explained": ev, "cumulative": cum})
    df_pca = pd.DataFrame(pca_rows)
    d_pca = X.shape[1]
    print("单成分解释方差（各向同性参照 = %.3f）:" % (1 / d_pca))
    print(df_pca.pivot(index="n_components", columns="seed", values="explained").round(3).to_string())
    df_pca.to_csv(out_path(args, "pca_scree.csv"), index=False)

    print(f"\n>>> 已保存到 {args.output_dir or paths.SHAPE + '/expN_*'}")
    print("\n>>> 提醒：功效分析只覆盖 (a) 高斯团块平移 与 (b) 弯曲流形两类另类假设。")
    print("    对未模拟的结构类型，本脚本的 null 结果不构成排除证据。")


if __name__ == "__main__":
    main()
