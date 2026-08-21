# -*- coding: utf-8 -*-
"""Train and evaluate DQN/PPO/SAC baselines against the existing CEM policy.

The comparison is intentionally run inside the same EnergyStorageEnv used by CEM:
same household CSVs, same reward/economics, same battery physics, same holdout
window, and deterministic evaluation.
"""

import argparse
import concurrent.futures
import glob
import json
import multiprocessing
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd

from energy_storage_env import (
    MAX_CHARGE_POWER_W,
    MAX_DISCHARGE_POWER_W,
    EnergyStorageEnv,
    apply_battery_physics,
    calculate_economics,
)


try:
    import torch
    from stable_baselines3 import DQN, PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        "Missing RL dependency: "
        f"{missing_package}. Install with `python -m pip install -r requirements.txt`."
    ) from exc


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: str
    cem_dir: str
    output_dir: str
    algorithms: list[str]
    households: list[str]
    seeds: list[int]
    timesteps: int
    sell_ratio: float
    val_len: int
    eval_len: int
    train_episode_len: int
    action_bins: int
    discrete_algorithms: list[str]
    algorithm_presets: dict[str, str]
    device: str
    gpu_ids: list[int]
    log_interval_steps: int
    rl_preset: str
    torch_num_threads: int
    workers: int
    checkpoint_interval_steps: int
    strict_torch_determinism: bool
    worker_log_dir: str


def format_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class EtaCallback(BaseCallback):
    """Print current training progress and ETA at fixed step intervals."""

    def __init__(
        self,
        total_timesteps: int,
        log_interval_steps: int,
        label: str,
    ):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.log_interval_steps = max(1, log_interval_steps)
        self.label = label
        self.start_time = 0.0
        self.last_log_step = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_log_step = 0
        print(f"        progress {self.label}: 0/{self.total_timesteps:,} steps", flush=True)

    def _on_step(self) -> bool:
        steps = min(int(self.num_timesteps), self.total_timesteps)
        if steps >= self.total_timesteps and self.last_log_step >= self.total_timesteps:
            return True
        if steps - self.last_log_step < self.log_interval_steps and steps < self.total_timesteps:
            return True

        elapsed = time.time() - self.start_time
        steps_per_sec = steps / elapsed if elapsed > 0 else 0.0
        remaining_steps = max(0, self.total_timesteps - steps)
        eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else float("inf")
        pct = 100.0 * steps / self.total_timesteps if self.total_timesteps else 100.0
        print(
            f"        progress {self.label}: {steps:,}/{self.total_timesteps:,} "
            f"({pct:5.1f}%) | {steps_per_sec:,.1f} steps/s | ETA {format_duration(eta)}"
            ,
            flush=True,
        )
        self.last_log_step = steps
        return True


class FileEtaCallback(EtaCallback):
    """ETA callback that writes progress to a per-job log file."""

    def __init__(
        self,
        total_timesteps: int,
        log_interval_steps: int,
        label: str,
        log_path: str,
    ):
        super().__init__(total_timesteps, log_interval_steps, label)
        self.log_path = log_path

    def _write(self, message: str) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_log_step = 0
        self._write(f"progress {self.label}: 0/{self.total_timesteps:,} steps")

    def _on_step(self) -> bool:
        steps = min(int(self.num_timesteps), self.total_timesteps)
        if steps >= self.total_timesteps and self.last_log_step >= self.total_timesteps:
            return True
        if steps - self.last_log_step < self.log_interval_steps and steps < self.total_timesteps:
            return True

        elapsed = time.time() - self.start_time
        steps_per_sec = steps / elapsed if elapsed > 0 else 0.0
        remaining_steps = max(0, self.total_timesteps - steps)
        eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else float("inf")
        pct = 100.0 * steps / self.total_timesteps if self.total_timesteps else 100.0
        self._write(
            f"progress {self.label}: {steps:,}/{self.total_timesteps:,} "
            f"({pct:5.1f}%) | {steps_per_sec:,.1f} steps/s | ETA {format_duration(eta)}"
        )
        self.last_log_step = steps
        return True


class SilentCallback(BaseCallback):
    """No-op callback for worker processes when per-job ETA would be noisy."""

    def _on_step(self) -> bool:
        return True


def build_callbacks(job: dict) -> BaseCallback:
    callbacks: list[BaseCallback] = []
    label = f"{job['algorithm']} household={job['household']} seed={job['seed']}"
    if job.get("show_progress", False):
        callbacks.append(
            EtaCallback(
                total_timesteps=job["timesteps"],
                log_interval_steps=job["log_interval_steps"],
                label=label,
            )
        )
    if job.get("worker_log_dir"):
        log_path = os.path.join(
            job["worker_log_dir"],
            f"{job['algorithm']}_{job['household']}_seed{job['seed']}.log",
        )
        callbacks.append(
            FileEtaCallback(
                total_timesteps=job["timesteps"],
                log_interval_steps=job["log_interval_steps"],
                label=label,
                log_path=log_path,
            )
        )
    if job["checkpoint_interval_steps"] > 0:
        checkpoint_dir = os.path.join(
            job["output_dir"],
            "checkpoints",
            job.get("method_name", job["algorithm"]),
            job["household"],
            f"seed_{job['seed']}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=job["checkpoint_interval_steps"],
                save_path=checkpoint_dir,
                name_prefix="model",
                save_replay_buffer=job["algorithm"] in {"dqn", "sac"},
                save_vecnormalize=False,
            )
        )
    if not callbacks:
        return SilentCallback()
    if len(callbacks) == 1:
        return callbacks[0]
    return CallbackList(callbacks)


class RandomStartEnergyStorageEnv(EnergyStorageEnv):
    """CEM-compatible env that samples training windows on reset."""

    def __init__(self, *args, random_starts: bool = False, rng_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_starts = random_starts
        self._rng = np.random.default_rng(rng_seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.random_starts and not (options and "start_idx" in options):
            max_start = max(1, self.total_data_steps - self.episode_length_hours - 24)
            options = {"start_idx": int(self._rng.integers(0, max_start))}
        return super().reset(seed=seed, options=options)


class DiscreteBatteryActionWrapper(gym.Wrapper):
    """Map a discrete action id to the env's normalized continuous action."""

    def __init__(self, env: gym.Env, bins: int = 21):
        super().__init__(env)
        if bins < 3 or bins % 2 == 0:
            raise ValueError("--action-bins must be an odd integer >= 3")
        self.actions = np.linspace(-1.0, 1.0, bins, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(bins)

    def step(self, action):
        normalized_action = float(self.actions[int(action)])
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([normalized_action], dtype=np.float32)
        )
        info["normalized_action"] = normalized_action
        return obs, reward, terminated, truncated, info


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_algorithm_presets(value: str) -> dict[str, str]:
    presets = {}
    for item in parse_csv_list(value):
        if "=" not in item:
            raise ValueError(
                "--algorithm-presets entries must use algorithm=preset, e.g. sac=fast"
            )
        algorithm, preset = [part.strip().lower() for part in item.split("=", 1)]
        if not algorithm or not preset:
            raise ValueError(
                "--algorithm-presets entries must use algorithm=preset, e.g. sac=fast"
            )
        presets[algorithm] = preset
    return presets


def split_household_df(csv_path: str, val_len: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if len(df) <= val_len + 24:
        raise ValueError(f"{csv_path} is too short for val_len={val_len}")
    return df.iloc[:-val_len].copy(), df.iloc[-val_len:].copy()


def uses_discrete_action_space(algorithm: str, discrete_algorithms: set[str]) -> bool:
    return algorithm == "dqn" or algorithm in discrete_algorithms


def method_name(algorithm: str, discrete_action_space: bool, rl_preset: str) -> str:
    preset_suffix = "" if rl_preset == "standard" else f"_{rl_preset}"
    if algorithm == "ppo" and discrete_action_space:
        return f"ppo_discrete{preset_suffix}"
    return f"{algorithm}{preset_suffix}"


def make_train_env(
    train_df: pd.DataFrame,
    algorithm: str,
    sell_ratio: float,
    episode_len: int,
    seed: int,
    action_bins: int,
    discrete_algorithms: set[str],
):
    env = RandomStartEnergyStorageEnv(
        train_df,
        episode_length_hours=episode_len,
        sell_price_ratio=sell_ratio,
        action_penalty_coef=0.0,
        meta_path=os.path.join(os.getcwd(), "data", "german_metadata.json"),
        random_starts=True,
        rng_seed=seed,
    )
    if uses_discrete_action_space(algorithm, discrete_algorithms):
        env = DiscreteBatteryActionWrapper(env, bins=action_bins)
    return Monitor(env)


def make_eval_env(
    eval_df: pd.DataFrame,
    algorithm: str,
    sell_ratio: float,
    eval_len: int,
    action_bins: int,
    discrete_algorithms: set[str],
):
    env = RandomStartEnergyStorageEnv(
        eval_df,
        episode_length_hours=eval_len,
        sell_price_ratio=sell_ratio,
        action_penalty_coef=0.0,
        meta_path=os.path.join(os.getcwd(), "data", "german_metadata.json"),
        random_starts=False,
    )
    if uses_discrete_action_space(algorithm, discrete_algorithms):
        env = DiscreteBatteryActionWrapper(env, bins=action_bins)
    return Monitor(env)


def build_model(algorithm: str, env: gym.Env, seed: int, device: str, rl_preset: str):
    if rl_preset not in {"fast", "standard", "heavy"}:
        raise ValueError(f"Unsupported --rl-preset: {rl_preset}")

    if algorithm == "dqn":
        if rl_preset == "heavy":
            return DQN(
                "MlpPolicy",
                env,
                seed=seed,
                device=device,
                verbose=0,
                learning_rate=1e-3,
                buffer_size=100_000,
                learning_starts=1_000,
                batch_size=256,
                gamma=0.99,
                train_freq=1,
                target_update_interval=1_000,
                exploration_fraction=0.15,
                exploration_final_eps=0.05,
                policy_kwargs={"net_arch": [128, 128]},
            )
        if rl_preset == "fast":
            return DQN(
                "MlpPolicy",
                env,
                seed=seed,
                device=device,
                verbose=0,
                learning_rate=1e-4,
                buffer_size=50_000,
                learning_starts=1_000,
                batch_size=32,
                gamma=0.99,
                train_freq=8,
                gradient_steps=1,
                target_update_interval=10_000,
                exploration_fraction=0.10,
                exploration_final_eps=0.05,
                policy_kwargs={"net_arch": [64, 64]},
            )
        return DQN(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=0,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=10_000,
            exploration_fraction=0.10,
            exploration_final_eps=0.05,
            policy_kwargs={"net_arch": [64, 64]},
        )
    if algorithm == "ppo":
        policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}
        if rl_preset == "standard":
            policy_kwargs = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
        if rl_preset == "fast":
            return PPO(
                "MlpPolicy",
                env,
                seed=seed,
                device=device,
                verbose=0,
                learning_rate=3e-4,
                n_steps=1_024,
                batch_size=256,
                n_epochs=5,
                gamma=0.99,
                gae_lambda=0.95,
                policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            )
        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2_048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs=policy_kwargs,
        )
    if algorithm == "sac":
        policy_kwargs = {"net_arch": [128, 128]}
        if rl_preset == "standard":
            policy_kwargs = {"net_arch": [64, 64]}
        if rl_preset == "fast":
            return SAC(
                "MlpPolicy",
                env,
                seed=seed,
                device=device,
                verbose=0,
                learning_rate=3e-4,
                buffer_size=50_000,
                learning_starts=5_000,
                batch_size=128,
                gamma=0.99,
                tau=0.005,
                train_freq=8,
                gradient_steps=1,
                ent_coef="auto",
                policy_kwargs={"net_arch": [64, 64]},
            )
        return SAC(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=0,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def evaluate_sb3_model(model, env: gym.Env, eval_len: int) -> dict:
    obs, _ = env.reset(options={"start_idx": 0})
    total_profit = 0.0
    total_reward = 0.0
    total_abs_action = 0.0
    final_soc = np.nan
    steps = 0
    for _ in range(eval_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_profit += float(info.get("profit", reward))
        normalized_action = info.get("normalized_action", action)
        total_abs_action += float(np.abs(np.asarray(normalized_action)).mean())
        final_soc = float(info.get("soc", np.nan))
        steps += 1
        if terminated or truncated:
            break
    return {
        "total_profit": total_profit,
        "avg_profit_eur_h": total_profit / steps,
        "total_reward": total_reward,
        "avg_abs_action": total_abs_action / steps,
        "final_soc": final_soc,
        "eval_steps": steps,
    }


def rollout_sb3_model(model, env: gym.Env, eval_len: int) -> pd.DataFrame:
    obs, _ = env.reset(options={"start_idx": 0})
    rows = []
    cumulative_profit = 0.0
    for step in range(eval_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        profit = float(info.get("profit", reward))
        cumulative_profit += profit
        normalized_action = float(np.asarray(info.get("normalized_action", action)).reshape(-1)[0])
        rows.append(
            {
                "step": step,
                "action_norm": normalized_action,
                "profit": profit,
                "cumulative_profit": cumulative_profit,
                "soc": float(info.get("soc", np.nan)),
            }
        )
        if terminated or truncated:
            break
    return pd.DataFrame(rows)


def evaluate_cem(weights_path: str, eval_df: pd.DataFrame, eval_len: int, sell_ratio: float) -> dict:
    env = EnergyStorageEnv(
        eval_df,
        episode_length_hours=eval_len,
        sell_price_ratio=sell_ratio,
        action_penalty_coef=0.0,
        meta_path=os.path.join(os.getcwd(), "data", "german_metadata.json"),
    )
    state, _ = env.reset(options={"start_idx": 0})
    total_profit = 0.0
    total_abs_action = 0.0
    final_soc = np.nan
    W = np.load(weights_path)
    steps = 0
    for _ in range(eval_len):
        action_val = float(np.tanh(np.dot(W[:-1], state) + W[-1]))
        state, _, done, _, info = env.step(np.array([action_val], dtype=np.float32))
        total_profit += float(info["profit"])
        total_abs_action += abs(action_val)
        final_soc = float(info["soc"])
        steps += 1
        if done:
            break
    env.close()
    return {
        "total_profit": total_profit,
        "avg_profit_eur_h": total_profit / steps,
        "total_reward": total_profit,
        "avg_abs_action": total_abs_action / steps,
        "final_soc": final_soc,
        "eval_steps": steps,
    }


def rollout_cem(weights_path: str, eval_df: pd.DataFrame, eval_len: int, sell_ratio: float) -> pd.DataFrame:
    env = EnergyStorageEnv(
        eval_df,
        episode_length_hours=eval_len,
        sell_price_ratio=sell_ratio,
        action_penalty_coef=0.0,
        meta_path=os.path.join(os.getcwd(), "data", "german_metadata.json"),
    )
    state, _ = env.reset(options={"start_idx": 0})
    W = np.load(weights_path)
    rows = []
    cumulative_profit = 0.0
    for step in range(eval_len):
        action_val = float(np.tanh(np.dot(W[:-1], state) + W[-1]))
        state, _, done, _, info = env.step(np.array([action_val], dtype=np.float32))
        profit = float(info["profit"])
        cumulative_profit += profit
        rows.append(
            {
                "step": step,
                "action_norm": action_val,
                "profit": profit,
                "cumulative_profit": cumulative_profit,
                "soc": float(info["soc"]),
            }
        )
        if done:
            break
    env.close()
    return pd.DataFrame(rows)


def evaluate_rule(
    eval_df: pd.DataFrame,
    eval_len: int,
    sell_ratio: float,
    policy_fn: Callable[[pd.Series, float], float],
) -> dict:
    ranked = eval_df.copy()
    ranked["_rank"] = ranked.groupby(ranked.index.date)["price_eur_kwh"].rank(pct=True)
    soc = 0.5
    total_profit = 0.0
    total_abs_action = 0.0
    for step in range(eval_len):
        row = ranked.iloc[step]
        intended_w = policy_fn(row, soc)
        soc, actual_port_w = apply_battery_physics(soc, intended_w)
        total_profit += calculate_economics(
            row["pv_gen_wh"],
            row["user_load_wh"],
            actual_port_w,
            row["price_eur_kwh"],
            sell_ratio,
        )
        total_abs_action += abs(intended_w) / MAX_CHARGE_POWER_W
    return {
        "total_profit": total_profit,
        "avg_profit_eur_h": total_profit / eval_len,
        "total_reward": total_profit,
        "avg_abs_action": total_abs_action / eval_len,
        "final_soc": soc,
        "eval_steps": eval_len,
    }


def greedy_policy(row: pd.Series, soc: float) -> float:
    del soc
    return float(row["pv_gen_wh"] - row["user_load_wh"])


def company_proxy_policy(row: pd.Series, soc: float) -> float:
    del soc
    price = float(row["price_eur_kwh"])
    rank = float(row["_rank"])
    if price <= 0.00244:
        return MAX_CHARGE_POWER_W
    if rank >= 0.75:
        return -MAX_DISCHARGE_POWER_W
    if rank <= 0.25:
        return MAX_CHARGE_POWER_W
    return 0.0


def no_battery_policy(row: pd.Series, soc: float) -> float:
    del row, soc
    return 0.0


def result_row(household: str, method: str, seed: int | str, metrics: dict) -> dict:
    row = {"household": household, "method": method, "seed": seed}
    row.update(metrics)
    return row


def set_global_seed(seed: int, strict_torch_determinism: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if strict_torch_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_gpu_ids(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_device(device_arg: str, gpu_ids: list[int]) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"--device {device_arg} was requested, but torch.cuda.is_available() is False."
        )
    if device_arg.startswith("cuda"):
        visible_count = torch.cuda.device_count()
        bad_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= visible_count]
        if bad_ids:
            raise SystemExit(
                f"--gpu-ids contains unavailable CUDA device ids {bad_ids}; "
                f"torch sees {visible_count} CUDA device(s)."
            )
    return device_arg


def discover_households(data_dir: str, cem_dir: str, limit: int | None) -> list[str]:
    csv_ids = {
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(data_dir, "*.csv"))
    }
    cem_ids = {
        os.path.basename(path).replace("_expert.npy", "")
        for path in glob.glob(os.path.join(cem_dir, "*_expert.npy"))
    }
    households = sorted(csv_ids & cem_ids)
    return households[:limit] if limit else households


def write_reproducibility_files(config: ExperimentConfig, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "stable_baselines3": sys.modules["stable_baselines3"].__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    with open(os.path.join(output_dir, "environment_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def run_rl_job(job: dict) -> dict:
    """Train/evaluate one algorithm-household-seed job.

    This function is process-safe: it owns its environments, model path, seeds,
    and output files.
    """
    if job["torch_num_threads"] > 0:
        torch.set_num_threads(job["torch_num_threads"])
    set_global_seed(job["seed"], job["strict_torch_determinism"])

    household = job["household"]
    algorithm = job["algorithm"]
    seed = job["seed"]
    csv_path = os.path.join(job["data_dir"], f"{household}.csv")
    train_df, eval_df = split_household_df(csv_path, job["val_len"])
    train_env = make_train_env(
        train_df,
        algorithm,
        job["sell_ratio"],
        job["train_episode_len"],
        seed,
        job["action_bins"],
        job["discrete_algorithms"],
    )
    eval_env = make_eval_env(
        eval_df,
        algorithm,
        job["sell_ratio"],
        job["eval_len"],
        job["action_bins"],
        job["discrete_algorithms"],
    )

    model_dir = os.path.join(job["output_dir"], "models", job["method_name"], household, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.zip")
    start = time.time()
    status = "trained"

    try:
        if job["skip_existing"] and os.path.exists(model_path):
            model_cls = {"dqn": DQN, "ppo": PPO, "sac": SAC}[algorithm]
            model = model_cls.load(model_path, env=eval_env, device=job["device"])
            status = "loaded"
        else:
            model = build_model(
                algorithm,
                train_env,
                seed,
                job["device"],
                job["rl_preset"],
            )
            callback = build_callbacks(job)
            model.learn(
                total_timesteps=job["timesteps"],
                callback=callback,
                progress_bar=False,
            )
            model.save(model_path)

        metrics = evaluate_sb3_model(model, eval_env, job["eval_len"])
        if job["save_rollouts"]:
            rollout_dir = os.path.join(job["output_dir"], "rollouts", job["method_name"], household)
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_path = os.path.join(rollout_dir, f"seed_{seed}.csv")
            rollout_sb3_model(model, eval_env, job["eval_len"]).to_csv(rollout_path, index=False)
        metrics["train_seconds"] = time.time() - start
        metrics["timesteps"] = job["timesteps"]
        metrics["status"] = status
        discrete_action_space = uses_discrete_action_space(algorithm, job["discrete_algorithms"])
        metrics["algorithm"] = algorithm
        metrics["rl_preset"] = job["rl_preset"]
        metrics["action_space"] = "discrete" if discrete_action_space else "continuous"
        metrics["action_bins"] = job["action_bins"] if discrete_action_space else np.nan
        row = result_row(household, job["method_name"], seed, metrics)
        return {"ok": True, "row": row, "job": job}
    except Exception as exc:
        return {
            "ok": False,
            "job": job,
            "error": repr(exc),
            "train_seconds": time.time() - start,
        }
    finally:
        train_env.close()
        eval_env.close()


def select_job_device(base_device: str, gpu_ids: list[int], job_index: int) -> str:
    if not base_device.startswith("cuda") or not gpu_ids:
        return base_device
    return f"cuda:{gpu_ids[job_index % len(gpu_ids)]}"


def with_job_device(job: dict, device: str) -> dict:
    assigned = dict(job)
    assigned["device"] = device
    return assigned


def job_action_label(job: dict) -> str:
    discrete_action_space = uses_discrete_action_space(
        job["algorithm"], job["discrete_algorithms"]
    )
    if discrete_action_space:
        return f"action=discrete({job['action_bins']} bins)"
    return "action=continuous"


def resolve_job_preset(args, algorithm: str) -> str:
    return args.algorithm_presets.get(algorithm, args.rl_preset)


def make_rl_jobs(args, households: list[str], algorithms: list[str], seeds: list[int]) -> list[dict]:
    jobs = []
    discrete_algorithms = set(args.discrete_algorithms)
    for household in households:
        for algorithm in algorithms:
            for seed in seeds:
                job_index = len(jobs)
                rl_preset = resolve_job_preset(args, algorithm)
                discrete_action_space = uses_discrete_action_space(algorithm, discrete_algorithms)
                jobs.append(
                    {
                        "household": household,
                        "algorithm": algorithm,
                        "method_name": method_name(algorithm, discrete_action_space, rl_preset),
                        "seed": seed,
                        "data_dir": args.data_dir,
                        "output_dir": args.output_dir,
                        "sell_ratio": args.ratio,
                        "val_len": args.val_len,
                        "eval_len": args.eval_len,
                        "train_episode_len": args.train_episode_len,
                        "action_bins": args.action_bins,
                        "discrete_algorithms": discrete_algorithms,
                        "device": select_job_device(args.device, args.gpu_ids, job_index),
                        "rl_preset": rl_preset,
                        "torch_num_threads": args.torch_num_threads,
                        "strict_torch_determinism": args.strict_torch_determinism,
                        "timesteps": args.timesteps,
                        "log_interval_steps": args.log_interval_steps,
                        "checkpoint_interval_steps": args.checkpoint_interval_steps,
                        "worker_log_dir": args.worker_log_dir,
                        "save_rollouts": args.save_rollouts,
                        "show_progress": args.workers == 1,
                        "skip_existing": args.skip_existing,
                    }
                )
    return jobs


def write_result_tables(rows: list[dict], output_dir: str) -> tuple[str, str]:
    results_path = os.path.join(output_dir, "comparison_results.csv")
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    df_results = pd.DataFrame(rows)
    df_results.to_csv(results_path, index=False)
    summary = (
        df_results.groupby("method")["avg_profit_eur_h"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    summary.to_csv(summary_path, index=False)
    return results_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN/PPO/SAC baselines and compare them with CEM on the same holdout windows."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cem-dir", default="results")
    parser.add_argument("--output-dir", default="results/rl_baselines")
    parser.add_argument("--algorithms", default="dqn,ppo,sac")
    parser.add_argument("--households", default="", help="Comma-separated IDs. Empty means all households with CEM vectors.")
    parser.add_argument("--limit-households", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--val-len", type=int, default=96)
    parser.add_argument("--eval-len", type=int, default=72)
    parser.add_argument("--train-episode-len", type=int, default=72)
    parser.add_argument(
        "--action-bins",
        type=int,
        default=None,
        help="Odd number of normalized action levels for discrete-action baselines, e.g. 11 or 21.",
    )
    parser.add_argument(
        "--dqn-action-bins",
        type=int,
        default=None,
        help="Deprecated alias for --action-bins, kept for older commands.",
    )
    parser.add_argument(
        "--discrete-algorithms",
        default="dqn",
        help="Comma-separated algorithms to run with the discrete action wrapper. DQN is always discrete; PPO can be added. SAC must remain continuous with SB3.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Comma-separated CUDA ids for round-robin job assignment, e.g. 0,1. Use with --device cuda.",
    )
    parser.add_argument(
        "--rl-preset",
        choices=["fast", "standard", "heavy"],
        default="standard",
        help="fast reduces NN update cost for screening; standard is the recommended fair baseline; heavy keeps earlier larger-network settings.",
    )
    parser.add_argument(
        "--algorithm-presets",
        default="",
        help="Optional comma-separated per-algorithm overrides, e.g. sac=fast. Unspecified algorithms use --rl-preset.",
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="CPU threads used by PyTorch. Small MLP RL training is often faster with 1-2 threads.",
    )
    parser.add_argument(
        "--strict-torch-determinism",
        action="store_true",
        help="Best-effort deterministic PyTorch kernels. Can be slower on GPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel RL training jobs. Keep small on CPU, e.g. 2-4.",
    )
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=10_000,
        help="Print per-model ETA every N training steps.",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=0,
        help="Save SB3 checkpoints every N training steps. 0 disables checkpoints.",
    )
    parser.add_argument(
        "--worker-log-dir",
        default="",
        help="Directory for per-job progress logs, useful when --workers > 1.",
    )
    parser.add_argument(
        "--save-rollouts",
        action="store_true",
        help="Save deterministic holdout action/SOC/profit trajectories for CEM and RL models.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.action_bins = (
        args.action_bins
        if args.action_bins is not None
        else args.dqn_action_bins
        if args.dqn_action_bins is not None
        else 21
    )
    if args.action_bins < 3 or args.action_bins % 2 == 0:
        raise SystemExit("--action-bins must be an odd integer >= 3 so that zero is included.")
    args.discrete_algorithms = {
        algorithm.lower() for algorithm in parse_csv_list(args.discrete_algorithms)
    }
    args.discrete_algorithms.add("dqn")
    if "sac" in args.discrete_algorithms:
        raise SystemExit(
            "SB3 SAC supports continuous Box action spaces only. "
            "Remove sac from --discrete-algorithms, or implement a separate Discrete SAC baseline."
        )
    try:
        args.algorithm_presets = parse_algorithm_presets(args.algorithm_presets)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    args.device = resolve_device(args.device, args.gpu_ids)
    if not args.worker_log_dir:
        args.worker_log_dir = os.path.join(args.output_dir, "worker_logs")
    if args.device.startswith("cuda") and args.workers > 1 and not args.gpu_ids:
        print(
            "WARNING: --device cuda with --workers > 1 will make multiple processes share the same GPU. "
            "For strict baseline runs, pass --gpu-ids, prefer --workers 1, or launch separate jobs with CUDA_VISIBLE_DEVICES.",
            flush=True,
        )
    if args.gpu_ids and args.workers > len(args.gpu_ids):
        print(
            f"WARNING: --workers {args.workers} is larger than --gpu-ids count {len(args.gpu_ids)}; "
            "some GPUs will run multiple jobs.",
            flush=True,
        )

    algorithms = [a.lower() for a in parse_csv_list(args.algorithms)]
    unsupported = sorted(set(algorithms) - {"dqn", "ppo", "sac"})
    if unsupported:
        raise SystemExit(f"Unsupported algorithms: {unsupported}")
    unsupported_discrete = sorted(args.discrete_algorithms - {"dqn", "ppo"})
    if unsupported_discrete:
        raise SystemExit(f"Unsupported --discrete-algorithms entries: {unsupported_discrete}")
    unsupported_preset_algorithms = sorted(set(args.algorithm_presets) - {"dqn", "ppo", "sac"})
    if unsupported_preset_algorithms:
        raise SystemExit(f"Unsupported --algorithm-presets algorithms: {unsupported_preset_algorithms}")
    unsupported_preset_values = {
        algorithm: preset
        for algorithm, preset in args.algorithm_presets.items()
        if preset not in {"fast", "standard", "heavy"}
    }
    if unsupported_preset_values:
        raise SystemExit(f"Unsupported --algorithm-presets values: {unsupported_preset_values}")

    if args.households:
        households = parse_csv_list(args.households)
    else:
        households = discover_households(args.data_dir, args.cem_dir, args.limit_households)
    seeds = [int(seed) for seed in parse_csv_list(args.seeds)]
    if not households:
        raise SystemExit("No households found. Check --data-dir and --cem-dir.")

    config = ExperimentConfig(
        data_dir=args.data_dir,
        cem_dir=args.cem_dir,
        output_dir=args.output_dir,
        algorithms=algorithms,
        households=households,
        seeds=seeds,
        timesteps=args.timesteps,
        sell_ratio=args.ratio,
        val_len=args.val_len,
        eval_len=args.eval_len,
        train_episode_len=args.train_episode_len,
        action_bins=args.action_bins,
        discrete_algorithms=sorted(args.discrete_algorithms),
        algorithm_presets=args.algorithm_presets,
        device=args.device,
        gpu_ids=args.gpu_ids,
        log_interval_steps=args.log_interval_steps,
        rl_preset=args.rl_preset,
        torch_num_threads=args.torch_num_threads,
        workers=args.workers,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        strict_torch_determinism=args.strict_torch_determinism,
        worker_log_dir=args.worker_log_dir,
    )
    write_reproducibility_files(config, args.output_dir)

    all_rows: list[dict] = []

    for household in households:
        csv_path = os.path.join(args.data_dir, f"{household}.csv")
        cem_path = os.path.join(args.cem_dir, f"{household}_expert.npy")
        print(f"\n>>> Household {household}", flush=True)
        train_df, eval_df = split_household_df(csv_path, args.val_len)

        deterministic_baselines = {
            "cem": evaluate_cem(cem_path, eval_df, args.eval_len, args.ratio),
            "greedy": evaluate_rule(eval_df, args.eval_len, args.ratio, greedy_policy),
            "company_proxy": evaluate_rule(eval_df, args.eval_len, args.ratio, company_proxy_policy),
            "no_battery": evaluate_rule(eval_df, args.eval_len, args.ratio, no_battery_policy),
        }
        for method, metrics in deterministic_baselines.items():
            metrics["algorithm"] = method
            metrics["rl_preset"] = "deterministic"
            metrics["action_space"] = "continuous" if method == "cem" else "rule"
            metrics["action_bins"] = np.nan
            all_rows.append(result_row(household, method, "deterministic", metrics))
            print(f"    {method:<14} {metrics['avg_profit_eur_h']:+.6f} EUR/h", flush=True)
        if args.save_rollouts:
            rollout_dir = os.path.join(args.output_dir, "rollouts", "cem", household)
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_cem(cem_path, eval_df, args.eval_len, args.ratio).to_csv(
                os.path.join(rollout_dir, "deterministic.csv"),
                index=False,
            )

    results_path, summary_path = write_result_tables(all_rows, args.output_dir)

    rl_jobs = make_rl_jobs(args, households, algorithms, seeds)
    total_training_jobs = len(rl_jobs)
    completed_training_jobs = 0
    overall_start = time.time()
    workers = max(1, args.workers)
    if workers == 1:
        print(f"\n>>> Starting {total_training_jobs} RL jobs serially", flush=True)
        for job in rl_jobs:
            print(
                f"    start {job['algorithm']:<4} household={job['household']} "
                f"seed={job['seed']} device={job['device']} {job_action_label(job)} "
                f"timesteps={job['timesteps']:,}",
                flush=True,
            )
            result = run_rl_job(job)
            completed_training_jobs += 1
            if not result["ok"]:
                print(
                    f"    ERROR {job['algorithm']} household={job['household']} seed={job['seed']}: "
                    f"{result['error']}",
                    flush=True,
                )
                continue
            row = result["row"]
            all_rows.append(row)
            elapsed_overall = time.time() - overall_start
            avg_job_seconds = elapsed_overall / completed_training_jobs
            remaining_jobs = total_training_jobs - completed_training_jobs
            overall_eta = avg_job_seconds * remaining_jobs
            print(
                f"    done {row['method']:<4} household={row['household']} seed={row['seed']} "
                f"device={job['device']} {job_action_label(job)} {row['status']:<7} "
                f"{row['avg_profit_eur_h']:+.6f} EUR/h | "
                f"job time {format_duration(row['train_seconds'])} | "
                f"{completed_training_jobs}/{total_training_jobs} complete | "
                f"overall ETA {format_duration(overall_eta)}",
                flush=True,
            )
            results_path, summary_path = write_result_tables(all_rows, args.output_dir)
    else:
        print(
            f"\n>>> Starting {total_training_jobs} RL jobs with {workers} worker processes",
            flush=True,
        )
        strict_gpu_scheduling = (
            args.device.startswith("cuda") and args.gpu_ids and workers <= len(args.gpu_ids)
        )
        if strict_gpu_scheduling:
            print(
                ">>> Strict GPU scheduling enabled: at most one active RL job per listed GPU.",
                flush=True,
            )
        multiprocessing_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing_context,
        ) as executor:
            if strict_gpu_scheduling:
                pending_jobs = iter(rl_jobs)
                future_to_job = {}
                for gpu_id in args.gpu_ids[:workers]:
                    try:
                        base_job = next(pending_jobs)
                    except StopIteration:
                        break
                    job = with_job_device(base_job, f"cuda:{gpu_id}")
                    print(
                        f"    queued {job['algorithm']:<4} household={job['household']} "
                        f"seed={job['seed']} device={job['device']} {job_action_label(job)} "
                        f"timesteps={job['timesteps']:,}",
                        flush=True,
                    )
                    future_to_job[executor.submit(run_rl_job, job)] = job

                while future_to_job:
                    done_futures, _ = concurrent.futures.wait(
                        future_to_job,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done_futures:
                        job = future_to_job.pop(future)
                        completed_training_jobs += 1
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = {
                                "ok": False,
                                "job": job,
                                "error": repr(exc),
                                "train_seconds": 0.0,
                            }

                        elapsed_overall = time.time() - overall_start
                        avg_job_seconds = elapsed_overall / completed_training_jobs
                        remaining_jobs = total_training_jobs - completed_training_jobs
                        overall_eta = avg_job_seconds * remaining_jobs

                        if not result["ok"]:
                            print(
                                f"    ERROR {job['algorithm']} household={job['household']} "
                                f"seed={job['seed']}: device={job['device']} {result['error']} | "
                                f"{completed_training_jobs}/{total_training_jobs} complete | "
                                f"overall ETA {format_duration(overall_eta)}",
                                flush=True,
                            )
                        else:
                            row = result["row"]
                            all_rows.append(row)
                            print(
                                f"    done {row['method']:<4} household={row['household']} "
                                f"seed={row['seed']} device={job['device']} {job_action_label(job)} "
                                f"{row['status']:<7} {row['avg_profit_eur_h']:+.6f} EUR/h | "
                                f"job time {format_duration(row['train_seconds'])} | "
                                f"{completed_training_jobs}/{total_training_jobs} complete | "
                                f"overall ETA {format_duration(overall_eta)}",
                                flush=True,
                            )
                            results_path, summary_path = write_result_tables(all_rows, args.output_dir)

                        try:
                            next_base_job = next(pending_jobs)
                        except StopIteration:
                            continue
                        next_job = with_job_device(next_base_job, job["device"])
                        print(
                            f"    queued {next_job['algorithm']:<4} household={next_job['household']} "
                            f"seed={next_job['seed']} device={next_job['device']} "
                            f"{job_action_label(next_job)} timesteps={next_job['timesteps']:,}",
                            flush=True,
                        )
                        future_to_job[executor.submit(run_rl_job, next_job)] = next_job
            else:
                for job in rl_jobs[:workers]:
                    print(
                        f"    queued {job['algorithm']:<4} household={job['household']} "
                        f"seed={job['seed']} device={job['device']} {job_action_label(job)} "
                        f"timesteps={job['timesteps']:,}",
                        flush=True,
                    )
                future_to_job = {executor.submit(run_rl_job, job): job for job in rl_jobs}
                for future in concurrent.futures.as_completed(future_to_job):
                    job = future_to_job[future]
                    completed_training_jobs += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"ok": False, "job": job, "error": repr(exc), "train_seconds": 0.0}

                    elapsed_overall = time.time() - overall_start
                    avg_job_seconds = elapsed_overall / completed_training_jobs
                    remaining_jobs = total_training_jobs - completed_training_jobs
                    overall_eta = avg_job_seconds * remaining_jobs

                    if not result["ok"]:
                        print(
                            f"    ERROR {job['algorithm']} household={job['household']} seed={job['seed']}: "
                            f"device={job['device']} {result['error']} | "
                            f"{completed_training_jobs}/{total_training_jobs} complete | "
                            f"overall ETA {format_duration(overall_eta)}",
                            flush=True,
                        )
                        continue

                    row = result["row"]
                    all_rows.append(row)
                    print(
                        f"    done {row['method']:<4} household={row['household']} seed={row['seed']} "
                        f"device={job['device']} {job_action_label(job)} {row['status']:<7} "
                        f"{row['avg_profit_eur_h']:+.6f} EUR/h | "
                        f"job time {format_duration(row['train_seconds'])} | "
                        f"{completed_training_jobs}/{total_training_jobs} complete | "
                        f"overall ETA {format_duration(overall_eta)}",
                        flush=True,
                    )
                    results_path, summary_path = write_result_tables(all_rows, args.output_dir)

    _, _ = write_result_tables(all_rows, args.output_dir)
    summary = pd.read_csv(summary_path)
    print(f"\n>>> Results: {results_path}")
    print(f">>> Summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
