# -*- coding: utf-8 -*-
"""共享 72 小时策略回测的观测矩阵构造与 rollout。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
sys.path.insert(0, paths.CODE)
from energy_storage_env import (MAX_CHARGE_POWER_W, MAX_DISCHARGE_POWER_W,  # noqa: E402
                                MAX_SOC, MIN_SOC,
                                apply_battery_physics, calculate_economics)

DATA_DIR = paths.DATA
WINDOW_H = 72
HOURS_PER_YEAR = 8760
SELL_RATIO = 1.0


def build_obs_matrix(df, meta, all_states, local_scaling):
    """预算 _get_obs 中与 SOC 无关的 34 维（索引 1..34），返回 (T, 35) 矩阵，第 0 列留给 SOC。"""
    n = len(df)
    M = np.zeros((n, 35), dtype=np.float64)

    def sc(col, name):
        s = local_scaling.get(name)
        v = df[col].to_numpy(dtype=np.float64)
        return (v - s["min"]) / (s["max"] - s["min"]) if s and s["max"] > s["min"] else np.zeros(n)

    M[:, 1] = sc("pv_gen_wh", "pv_gen_wh")
    M[:, 2] = sc("user_load_wh", "user_load_wh")
    st = df["state"].iloc[0]
    M[:, 3] = (all_states.index(st) / max(1, len(all_states) - 1)) if st in all_states else 0.0
    h = 2 * np.pi * df["hour_of_day"].to_numpy(dtype=np.float64) / 24
    M[:, 4], M[:, 5] = np.sin(h), np.cos(h)
    g_min, g_max = meta["price_min"], meta["price_max"]
    M[:, 6] = (df["price_eur_kwh"].to_numpy(dtype=np.float64) - g_min) / (g_max - g_min + 1e-9)
    M[:, 7] = sc("price_eur_kwh", "price_eur_kwh")
    for i in range(1, 25):
        M[:, 7 + i] = sc(f"price_eur_kwh_t+{i}", f"price_eur_kwh_t+{i}")
    M[:, 32] = df["cloud_cover"].to_numpy(dtype=np.float64) / 100.0
    M[:, 33] = sc("shortwave_radiation", "shortwave_radiation")
    M[:, 34] = sc("shortwave_radiation_t+3", "shortwave_radiation_t+3")
    return M


def rollout(theta, M, pv, load, price, starts):
    """在给定窗口起点上跑 θ，返回总欧元利润与总步数。"""
    w, b = theta[:35], theta[35]
    total, steps = 0.0, 0
    for s0 in starts:
        soc = (MIN_SOC + MAX_SOC) / 2
        for t in range(s0, s0 + WINDOW_H):
            M[t, 0] = soc
            a = np.tanh(M[t] @ w + b)
            intended = a * (MAX_CHARGE_POWER_W if a > 0 else MAX_DISCHARGE_POWER_W)
            soc, port = apply_battery_physics(soc, intended)
            total += calculate_economics(pv[t], load[t], port, price[t], SELL_RATIO)
            steps += 1
    return total, steps
