"""Shared common-input battery replay; numerical kernel unchanged."""
import numpy as np
from gps_context import tail, env, build_obs_matrix, WINDOW_H

def replay(theta, obs, *, observe=None):
    """All strategies see identical external inputs; each has its own evolving SOC."""
    soc = np.full(len(theta), (env.MIN_SOC + env.MAX_SOC) / 2)
    drive = theta[:, 1:35] @ obs[:, 1:].T + theta[:, 35, None]
    ports = np.empty_like(drive)
    for t in range(len(obs)):
        a = np.tanh(drive[:, t] + theta[:, 0] * soc)
        intended = a * np.where(a > 0, env.MAX_CHARGE_POWER_W, env.MAX_DISCHARGE_POWER_W)
        upper = (env.MAX_SOC - soc) * env.BATTERY_CAPACITY_WH / env.CHARGE_EFFICIENCY
        lower = (env.MIN_SOC - soc) * env.BATTERY_CAPACITY_WH * env.DISCHARGE_EFFICIENCY
        p = np.clip(intended, lower, upper)
        if observe is not None:
            observe(t, soc, intended, p)
        if t == 0 or t == len(obs)-1:
            _, ref = env.apply_battery_physics(soc[0], intended[0])
            np.testing.assert_allclose(p[0], ref, atol=1e-7)
        soc = np.clip(soc + np.where(p > 0, p * env.CHARGE_EFFICIENCY,
                                    p / env.DISCHARGE_EFFICIENCY) / env.BATTERY_CAPACITY_WH,
                      env.MIN_SOC, env.MAX_SOC)
        ports[:, t] = p / 1000
    return ports
