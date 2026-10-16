# -*- coding: utf-8 -*-
"""
检验观察到的尾部方向能否超过保留家庭对应关系的高斯零参照，并比较沿该方向与随机方向
扰动策略时的收益变化。方向由同一批向量事后选出，所有结果均属探索性分析。零参照的
协方差分解与半正定投影见 `part_a()`。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_基础"))
import paths
sys.path.insert(0, paths.CODE)
import argparse
import glob
import itertools
import warnings
import zlib

import numpy as np

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from backtest_common import (DATA_DIR, HOURS_PER_YEAR, SELL_RATIO,  # noqa: E402
                             WINDOW_H, build_obs_matrix, rollout)
from energy_storage_env import EnergyStorageEnv  # noqa: E402


warnings.filterwarnings("ignore")

MIN_CLUSTER_SIZE = 4
MIN_SAMPLES = 2
N_REP = 300
N_EVAL_WIN = 8
N_RANDOM_DIR = 20
DELTA_SCALE = [0.5, 1.0, 2.0]   # 相对于"成员质心 - 主体质心"的实际幅度


def load_panel():
    seeds = sorted(d for d in os.listdir(paths.SEED_ROOT)
                   if glob.glob(os.path.join(paths.SEED_ROOT, d, "*_expert.npy")))
    per = {}
    for s in seeds:
        files = sorted(glob.glob(os.path.join(paths.SEED_ROOT, s, "*_expert.npy")))
        ids = [os.path.basename(f).replace("_expert.npy", "") for f in files]
        per[s] = (ids, np.array([np.load(f) for f in files]))
    ids = per[seeds[0]][0]
    keep = paths.dims_geometry(per[seeds[0]][1].shape[1])
    raw = {s: per[s][1][:, keep] for s in seeds}
    scaler = StandardScaler().fit(np.vstack([raw[s] for s in seeds]))
    return ids, seeds, raw, {s: scaler.transform(raw[s]) for s in seeds}, scaler, keep


def separating_direction(M):
    """HDBSCAN 的少数簇质心 - 主体质心，归一化；返回 (成员下标, 单位方向, 原始幅度)。"""
    lab = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES).fit_predict(M)
    real = [c for c in set(lab) if c != -1]
    if not real:
        return None, None, None
    sm = min(real, key=lambda c: (lab == c).sum())
    mem = np.where(lab == sm)[0]
    oth = np.setdiff1d(np.arange(len(M)), mem)
    if len(mem) < 3 or len(oth) < 3:
        return None, None, None
    d = M[mem].mean(axis=0) - M[oth].mean(axis=0)
    nrm = np.linalg.norm(d)
    return (mem, d / nrm, nrm) if nrm > 0 else (None, None, None)


def psd_project(C):
    w, V = np.linalg.eigh((C + C.T) / 2)
    return V @ np.diag(np.clip(w, 0, None)) @ V.T


def part_a(ids, seeds, Z, n_rep, rng):
    print("=" * 78)
    print(">>> (A) 保留家庭身份的零参照：方向一致性还超出吗")
    print("=" * 78)
    n, d, k = len(ids), Z[seeds[0]].shape[1], len(seeds)

    W = {}
    for s in seeds:
        _, W[s], _ = separating_direction(Z[s])
    obs = {(a, b): float(W[a] @ W[b]) for a, b in itertools.combinations(seeds, 2)}

    # 方差分解
    S = np.stack([Z[s] for s in seeds], axis=1)              # (n, k, d)
    hm = S.mean(axis=1)
    dev = (S - hm[:, None, :]).reshape(-1, d)
    cov_w = dev.T @ dev / (n * (k - 1))
    cov_b = psd_project(np.cov(hm, rowvar=False) - cov_w / k)
    mu = S.reshape(-1, d).mean(axis=0)
    icc_null = np.trace(cov_b) / (np.trace(cov_b) + np.trace(cov_w))
    print(f"  零参照按构造的 ICC = {icc_null:.3f}（真实数据 0.212，二者应接近）")

    null = []
    for _ in range(n_rep):
        z = rng.multivariate_normal(mu, cov_b, size=n)
        ws = []
        for _ in range(k):
            M = z + rng.multivariate_normal(np.zeros(d), cov_w, size=n)
            _, w, _ = separating_direction(M)
            if w is not None:
                ws.append(w)
        if len(ws) == k:
            null += [float(a @ b) for a, b in itertools.combinations(ws, 2)]
    null = np.array(null)
    print(f"  零参照 {len(null)} 个配对余弦：均值 {null.mean():+.3f}，"
          f"95% 分位 {np.quantile(null, .95):+.3f}，最大 {null.max():+.3f}")
    rows = []
    for (a, b), c in obs.items():
        p = (1 + (null >= c).sum()) / (1 + len(null))
        print(f"  观测 {a[-2:]} vs {b[-2:]} = {c:+.3f}   p = {p:.4f}"
              f"   {'仍超出零参照' if p < 0.05 else '不再超出'}")
        rows.append({"seed_a": a, "seed_b": b, "cos_obs": c, "p_identity_null": p,
                     "null_mean": null.mean(), "null_q95": np.quantile(null, .95),
                     "null_max": null.max(), "icc_null": icc_null})
    return pd.DataFrame(rows), pd.DataFrame({"null_cos": null}), W


def part_b(ids, seeds, raw, Z, scaler, keep, W, rng, n_rand):
    print("\n" + "=" * 78)
    print(">>> (B) 沿尾巴方向扰动 θ vs 沿随机方向：收益变化多少")
    print("=" * 78)
    probe = EnergyStorageEnv(pd.read_csv(os.path.join(DATA_DIR, f"{ids[0]}.csv"),
                                         index_col=0, parse_dates=True),
                             sell_price_ratio=SELL_RATIO)
    meta, all_states = probe.meta, probe.all_states
    scale = scaler.scale_
    n_dim_full = 36

    def expand(vec_keep):
        """把 keep 子空间上的增量放回完整 36 维（被排除的维增量为 0）。"""
        full = np.zeros(n_dim_full)
        full[keep] = vec_keep
        return full

    rows = []
    for idx, h in enumerate(ids, 1):
        df = pd.read_csv(os.path.join(DATA_DIR, f"{h}.csv"), index_col=0, parse_dates=True)
        env = EnergyStorageEnv(df, sell_price_ratio=SELL_RATIO)
        M = build_obs_matrix(df, meta, all_states, env.local_scaling)
        pv = df["pv_gen_wh"].to_numpy(float)
        ld = df["user_load_wh"].to_numpy(float)
        pr = df["price_eur_kwh"].to_numpy(float)
        wr = np.random.default_rng(zlib.crc32(h.encode("utf-8")))
        wins = wr.integers(0, len(df) - WINDOW_H - 1, size=N_EVAL_WIN)

        def val(theta_full):
            t, st = rollout(theta_full, M, pv, ld, pr, wins)
            return t / st * HOURS_PER_YEAR

        for s in seeds:
            th0 = np.load(os.path.join(paths.SEED_ROOT, s, f"{h}_expert.npy"))
            base = val(th0)
            _, wdir, mag = separating_direction(Z[s])
            for fac in DELTA_SCALE:
                delta = fac * mag
                dt = expand(delta * wdir * scale)
                tail = [abs(val(th0 + dt) - base), abs(val(th0 - dt) - base)]
                rnd = []
                for _ in range(n_rand):
                    r = rng.standard_normal(len(keep))
                    r /= np.linalg.norm(r)
                    rnd.append(abs(val(th0 + expand(delta * r * scale)) - base))
                rows.append({"household": h, "seed": s, "delta_factor": fac,
                             "delta_norm_std": delta, "base_eur_yr": base,
                             "abs_change_tail": float(np.mean(tail)),
                             "abs_change_random": float(np.mean(rnd))})
        if idx % 20 == 0 or idx == len(ids):
            print(f"    {idx}/{len(ids)} 户完成")

    D = pd.DataFrame(rows)
    print(f"\n  {'扰动幅度':<12}{'沿尾巴 |Δ收益|':>16}{'沿随机 |Δ收益|':>16}"
          f"{'尾巴/随机':>11}{'尾巴更平的户次':>15}")
    for fac, g in D.groupby("delta_factor"):
        t, r = g.abs_change_tail.median(), g.abs_change_random.median()
        flat = int((g.abs_change_tail < g.abs_change_random).sum())
        print(f"  ×{fac:<11}{t:>16.2f}{r:>16.2f}{t/r:>11.2f}{flat:>11}/{len(g)}")
    print("\n  比值 <1 = 沿尾巴方向更平（该方向对收益影响更小，支持'无代价漂移'）")
    print("  比值 >1 = 沿尾巴方向更陡（该方向是真实的策略权衡轴）")
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rep", type=int, default=N_REP)
    ap.add_argument("--n_random_dir", type=int, default=N_RANDOM_DIR)
    ap.add_argument("--skip_b", action="store_true")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    out = args.output_dir or os.path.join(
        paths.SHAPE, "exp24_tail_direction_validation")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(42)

    ids, seeds, raw, Z, scaler, keep = load_panel()
    print(f">>> {len(ids)} 户 × {len(seeds)} 重训 × {Z[seeds[0]].shape[1]} 维"
          f"\n")

    A, Anull, W = part_a(ids, seeds, Z, args.n_rep, rng)
    A.to_csv(os.path.join(out, "direction_identity_null.csv"), index=False)
    Anull.to_csv(os.path.join(out, "identity_null_cos_distribution.csv"), index=False)

    if not args.skip_b:
        B = part_b(ids, seeds, raw, Z, scaler, keep, W, rng, args.n_random_dir)
        B.to_csv(os.path.join(out, "perturbation_tail_vs_random.csv"), index=False)
    print(f"\n>>> 已保存到 {out}")


if __name__ == "__main__":
    main()
