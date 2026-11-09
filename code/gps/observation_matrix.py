"""Observation construction, copied unchanged from the existing backtest kernel."""
import numpy as np

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
