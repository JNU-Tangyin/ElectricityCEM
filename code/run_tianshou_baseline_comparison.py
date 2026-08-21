# -*- coding: utf-8 -*-
"""Train Tianshou DQN/PPO/Discrete-SAC baselines against the existing CEM policy.

All learning baselines use the same discrete battery action grid. The evaluation
protocol intentionally matches CEM: same household CSVs, same holdout window,
same battery physics, same economics, and deterministic holdout rollouts.
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
import torch
from torch import nn
from torch.distributions import Categorical

from energy_storage_env import (
    MAX_CHARGE_POWER_W,
    MAX_DISCHARGE_POWER_W,
    EnergyStorageEnv,
    apply_battery_physics,
    calculate_economics,
)


try:
    import tianshou
    from tianshou.data import Batch, ReplayBuffer, to_numpy
    from tianshou.policy import DQNPolicy, DiscreteSACPolicy, PPOPolicy
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        "Missing Tianshou dependency: "
        f"{missing_package}. Install with `python -m pip install tianshou==0.5.1`."
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
    device: str
    gpu_ids: list[int]
    workers: int
    torch_num_threads: int
    log_interval_steps: int
    buffer_size: int
    batch_size: int
    gamma: float
    hidden_sizes: list[int]
    dqn_train_freq: int
    dqn_gradient_steps: int
    dqn_learning_starts: int
    dqn_target_update_freq: int
    dqn_exploration_fraction: float
    dqn_final_eps: float
    ppo_rollout_steps: int
    ppo_batch_size: int
    ppo_epochs: int
    sac_train_freq: int
    sac_gradient_steps: int
    sac_learning_starts: int
    sac_alpha: float
    sac_tau: float
    strict_torch_determinism: bool


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


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def split_household_df(csv_path: str, val_len: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if len(df) <= val_len + 24:
        raise ValueError(f"{csv_path} is too short for val_len={val_len}")
    return df.iloc[:-val_len].copy(), df.iloc[-val_len:].copy()


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

    def __init__(self, env: gym.Env, bins: int):
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


def make_env(
    df: pd.DataFrame,
    sell_ratio: float,
    episode_len: int,
    action_bins: int,
    random_starts: bool,
    seed: int = 0,
) -> gym.Env:
    env = RandomStartEnergyStorageEnv(
        df,
        episode_length_hours=episode_len,
        sell_price_ratio=sell_ratio,
        action_penalty_coef=0.0,
        meta_path=os.path.join(os.getcwd(), "data", "german_metadata.json"),
        random_starts=random_starts,
        rng_seed=seed,
    )
    return DiscreteBatteryActionWrapper(env, bins=action_bins)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            obs = obs.to(next(self.parameters()).device).float()
        return self.net(obs)


class TianshouActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.net = MLP(obs_dim, action_dim, hidden_sizes)

    def forward(self, obs, state=None, info=None):
        del info
        return self.net(obs), state


class TianshouQNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.net = MLP(obs_dim, action_dim, hidden_sizes)

    def forward(self, obs, state=None, info=None):
        del info
        return self.net(obs), state


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.net = MLP(obs_dim, 1, hidden_sizes)

    def forward(self, obs):
        return self.net(obs)


class DiscreteCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.net = MLP(obs_dim, action_dim, hidden_sizes)

    def forward(self, obs):
        return self.net(obs)


def categorical_logits(logits):
    return Categorical(logits=logits)


def build_policy(
    algorithm: str,
    obs_dim: int,
    action_dim: int,
    hidden_sizes: list[int],
    device: str,
    args,
):
    action_space = gym.spaces.Discrete(action_dim)
    if algorithm == "dqn":
        model = TianshouQNet(obs_dim, action_dim, hidden_sizes).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.dqn_lr)
        policy = DQNPolicy(
            model,
            optim,
            discount_factor=args.gamma,
            estimation_step=1,
            target_update_freq=args.dqn_target_update_freq,
        )
        return policy
    if algorithm == "ppo":
        actor = TianshouActor(obs_dim, action_dim, hidden_sizes).to(device)
        critic = ValueNet(obs_dim, hidden_sizes).to(device)
        optim = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=args.ppo_lr,
        )
        policy = PPOPolicy(
            actor,
            critic,
            optim,
            categorical_logits,
            discount_factor=args.gamma,
            gae_lambda=args.ppo_gae_lambda,
            eps_clip=args.ppo_clip,
            vf_coef=args.ppo_vf_coef,
            ent_coef=args.ppo_ent_coef,
            max_grad_norm=args.ppo_max_grad_norm,
            action_space=action_space,
            action_scaling=False,
            deterministic_eval=True,
        )
        return policy
    if algorithm == "sac":
        actor = TianshouActor(obs_dim, action_dim, hidden_sizes).to(device)
        critic1 = DiscreteCritic(obs_dim, action_dim, hidden_sizes).to(device)
        critic2 = DiscreteCritic(obs_dim, action_dim, hidden_sizes).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.sac_lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.sac_lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.sac_lr)
        policy = DiscreteSACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=args.sac_tau,
            gamma=args.gamma,
            alpha=args.sac_alpha,
            estimation_step=1,
            deterministic_eval=True,
            action_space=action_space,
        )
        return policy
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def batch_for_obs(obs: np.ndarray) -> Batch:
    return Batch(obs=np.expand_dims(obs, axis=0), info=np.array([{}], dtype=object))


def policy_sample_action(policy, obs: np.ndarray) -> int:
    with torch.no_grad():
        out = policy(batch_for_obs(obs))
    return int(np.asarray(to_numpy(out.act)).reshape(-1)[0])


def deterministic_action(policy, algorithm: str, obs: np.ndarray) -> int:
    obs_batch = np.expand_dims(obs, axis=0)
    with torch.no_grad():
        if algorithm == "dqn":
            q_values, _ = policy.model(obs_batch)
            return int(torch.argmax(q_values, dim=-1).item())
        if algorithm in {"ppo", "sac"}:
            logits, _ = policy.actor(obs_batch)
            return int(torch.argmax(logits, dim=-1).item())
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def add_transition(buffer: ReplayBuffer, obs, act, rew, terminated, truncated, obs_next) -> None:
    buffer.add(
        Batch(
            obs=np.expand_dims(obs, axis=0),
            act=np.array([act], dtype=np.int64),
            rew=np.array([rew], dtype=np.float32),
            terminated=np.array([terminated], dtype=bool),
            truncated=np.array([truncated], dtype=bool),
            obs_next=np.expand_dims(obs_next, axis=0),
            info=np.array([{}], dtype=object),
        ),
        buffer_ids=[0],
    )


def dqn_epsilon(step: int, total_timesteps: int, args) -> float:
    frac_steps = max(1, int(total_timesteps * args.dqn_exploration_fraction))
    progress = min(1.0, step / frac_steps)
    return 1.0 + progress * (args.dqn_final_eps - 1.0)


def train_policy(policy, env: gym.Env, algorithm: str, seed: int, args, label: str) -> None:
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    start_time = time.time()
    last_log_step = 0
    print(f"        progress {label}: 0/{args.timesteps:,} steps", flush=True)

    if algorithm == "ppo":
        buffer = ReplayBuffer(args.ppo_rollout_steps)
    else:
        buffer = ReplayBuffer(args.buffer_size)

    policy.train()
    for step in range(1, args.timesteps + 1):
        if algorithm == "dqn" and rng.random() < dqn_epsilon(step, args.timesteps, args):
            action = int(rng.integers(0, env.action_space.n))
        else:
            action = policy_sample_action(policy, obs)

        obs_next, reward, terminated, truncated, _ = env.step(action)
        add_transition(buffer, obs, action, reward, terminated, truncated, obs_next)
        obs = obs_next
        if terminated or truncated:
            obs, _ = env.reset()

        if algorithm == "ppo":
            if len(buffer) >= args.ppo_rollout_steps:
                policy.update(
                    0,
                    buffer,
                    batch_size=args.ppo_batch_size,
                    repeat=args.ppo_epochs,
                )
                buffer.reset()
        else:
            train_freq = args.dqn_train_freq if algorithm == "dqn" else args.sac_train_freq
            gradient_steps = args.dqn_gradient_steps if algorithm == "dqn" else args.sac_gradient_steps
            learning_starts = args.dqn_learning_starts if algorithm == "dqn" else args.sac_learning_starts
            batch_size = args.batch_size
            if step >= learning_starts and len(buffer) >= batch_size and step % train_freq == 0:
                for _ in range(gradient_steps):
                    policy.update(batch_size, buffer)

        if step - last_log_step >= args.log_interval_steps or step == args.timesteps:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0.0
            remaining = args.timesteps - step
            eta = remaining / steps_per_sec if steps_per_sec > 0 else float("inf")
            pct = 100.0 * step / args.timesteps
            print(
                f"        progress {label}: {step:,}/{args.timesteps:,} "
                f"({pct:5.1f}%) | {steps_per_sec:,.1f} steps/s | ETA {format_duration(eta)}",
                flush=True,
            )
            last_log_step = step

    if algorithm == "ppo" and len(buffer) > 0:
        policy.update(0, buffer, batch_size=args.ppo_batch_size, repeat=args.ppo_epochs)


def evaluate_tianshou_policy(policy, algorithm: str, env: gym.Env, eval_len: int) -> dict:
    policy.eval()
    obs, _ = env.reset(options={"start_idx": 0})
    total_profit = 0.0
    total_reward = 0.0
    total_abs_action = 0.0
    final_soc = np.nan
    steps = 0
    for _ in range(eval_len):
        action = deterministic_action(policy, algorithm, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_profit += float(info.get("profit", reward))
        total_abs_action += abs(float(info.get("normalized_action", 0.0)))
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
    weights = np.load(weights_path)
    steps = 0
    for _ in range(eval_len):
        action_val = float(np.tanh(np.dot(weights[:-1], state) + weights[-1]))
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


def method_name(algorithm: str, action_bins: int) -> str:
    return f"{algorithm}_discrete_{action_bins}"


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
        "tianshou": tianshou.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "torch_num_threads": torch.get_num_threads(),
    }
    with open(os.path.join(output_dir, "environment_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


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


def run_rl_job(job: dict) -> dict:
    if job["torch_num_threads"] > 0:
        torch.set_num_threads(job["torch_num_threads"])
    set_global_seed(job["seed"], job["strict_torch_determinism"])

    household = job["household"]
    algorithm = job["algorithm"]
    seed = job["seed"]
    csv_path = os.path.join(job["data_dir"], f"{household}.csv")
    train_df, eval_df = split_household_df(csv_path, job["val_len"])
    train_env = make_env(
        train_df,
        job["sell_ratio"],
        job["train_episode_len"],
        job["action_bins"],
        random_starts=True,
        seed=seed,
    )
    eval_env = make_env(
        eval_df,
        job["sell_ratio"],
        job["eval_len"],
        job["action_bins"],
        random_starts=False,
        seed=seed,
    )

    obs_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(train_env.action_space.n)
    policy = build_policy(algorithm, obs_dim, action_dim, job["hidden_sizes"], job["device"], job["args"])

    model_dir = os.path.join(job["output_dir"], "models", job["method_name"], household, f"seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "policy.pt")
    start = time.time()
    status = "trained"

    try:
        if job["skip_existing"] and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=job["device"])
            policy.load_state_dict(checkpoint["policy_state_dict"])
            status = "loaded"
        else:
            train_policy(policy, train_env, algorithm, seed, job["args"], job["label"])
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "algorithm": algorithm,
                    "action_bins": job["action_bins"],
                    "hidden_sizes": job["hidden_sizes"],
                },
                model_path,
            )

        metrics = evaluate_tianshou_policy(policy, algorithm, eval_env, job["eval_len"])
        metrics["train_seconds"] = time.time() - start
        metrics["timesteps"] = job["timesteps"]
        metrics["status"] = status
        metrics["algorithm"] = algorithm
        metrics["library"] = "tianshou"
        metrics["action_space"] = "discrete"
        metrics["action_bins"] = job["action_bins"]
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


def make_rl_jobs(args, households: list[str], algorithms: list[str], seeds: list[int]) -> list[dict]:
    jobs = []
    for household in households:
        for algorithm in algorithms:
            for seed in seeds:
                job_index = len(jobs)
                job_args = argparse.Namespace(**vars(args))
                job_args.device = select_job_device(args.device, args.gpu_ids, job_index)
                jobs.append(
                    {
                        "household": household,
                        "algorithm": algorithm,
                        "method_name": method_name(algorithm, args.action_bins),
                        "seed": seed,
                        "data_dir": args.data_dir,
                        "output_dir": args.output_dir,
                        "sell_ratio": args.ratio,
                        "val_len": args.val_len,
                        "eval_len": args.eval_len,
                        "train_episode_len": args.train_episode_len,
                        "action_bins": args.action_bins,
                        "device": job_args.device,
                        "hidden_sizes": args.hidden_sizes,
                        "torch_num_threads": args.torch_num_threads,
                        "strict_torch_determinism": args.strict_torch_determinism,
                        "timesteps": args.timesteps,
                        "skip_existing": args.skip_existing,
                        "args": job_args,
                        "label": f"{algorithm}_discrete_{args.action_bins} household={household} seed={seed}",
                    }
                )
    return jobs


def print_job_done(row: dict, job: dict, completed: int, total: int, overall_eta: float) -> None:
    print(
        f"    done {row['method']:<16} household={row['household']} seed={row['seed']} "
        f"device={job['device']} {row['status']:<7} {row['avg_profit_eur_h']:+.6f} EUR/h | "
        f"job time {format_duration(row['train_seconds'])} | "
        f"{completed}/{total} complete | overall ETA {format_duration(overall_eta)}",
        flush=True,
    )


def print_job_error(job: dict, error: str, completed: int, total: int, overall_eta: float) -> None:
    print(
        f"    ERROR {job['algorithm']} household={job['household']} seed={job['seed']}: "
        f"device={job['device']} {error} | {completed}/{total} complete | "
        f"overall ETA {format_duration(overall_eta)}",
        flush=True,
    )


def handle_finished_result(result, job, all_rows, completed, total, overall_start, output_dir):
    elapsed = time.time() - overall_start
    avg_job_seconds = elapsed / completed
    overall_eta = avg_job_seconds * (total - completed)
    if not result["ok"]:
        print_job_error(job, result["error"], completed, total, overall_eta)
        return None
    row = result["row"]
    all_rows.append(row)
    print_job_done(row, job, completed, total, overall_eta)
    return write_result_tables(all_rows, output_dir)


def run_training_jobs(args, rl_jobs: list[dict], all_rows: list[dict]) -> None:
    total = len(rl_jobs)
    completed = 0
    overall_start = time.time()
    workers = max(1, args.workers)
    if workers == 1:
        print(f"\n>>> Starting {total} Tianshou RL jobs serially", flush=True)
        for job in rl_jobs:
            print(
                f"    start {job['method_name']:<16} household={job['household']} "
                f"seed={job['seed']} device={job['device']} timesteps={job['timesteps']:,}",
                flush=True,
            )
            result = run_rl_job(job)
            completed += 1
            handle_finished_result(result, job, all_rows, completed, total, overall_start, args.output_dir)
        return

    print(f"\n>>> Starting {total} Tianshou RL jobs with {workers} worker processes", flush=True)
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
                job["args"].device = job["device"]
                print(
                    f"    queued {job['method_name']:<16} household={job['household']} "
                    f"seed={job['seed']} device={job['device']} timesteps={job['timesteps']:,}",
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
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"ok": False, "job": job, "error": repr(exc), "train_seconds": 0.0}
                    handle_finished_result(result, job, all_rows, completed, total, overall_start, args.output_dir)

                    try:
                        next_base_job = next(pending_jobs)
                    except StopIteration:
                        continue
                    next_job = with_job_device(next_base_job, job["device"])
                    next_job["args"].device = next_job["device"]
                    print(
                        f"    queued {next_job['method_name']:<16} household={next_job['household']} "
                        f"seed={next_job['seed']} device={next_job['device']} "
                        f"timesteps={next_job['timesteps']:,}",
                        flush=True,
                    )
                    future_to_job[executor.submit(run_rl_job, next_job)] = next_job
        else:
            for job in rl_jobs[:workers]:
                print(
                    f"    queued {job['method_name']:<16} household={job['household']} "
                    f"seed={job['seed']} device={job['device']} timesteps={job['timesteps']:,}",
                    flush=True,
                )
            future_to_job = {executor.submit(run_rl_job, job): job for job in rl_jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"ok": False, "job": job, "error": repr(exc), "train_seconds": 0.0}
                handle_finished_result(result, job, all_rows, completed, total, overall_start, args.output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Train Tianshou DQN/PPO/Discrete-SAC baselines and compare with CEM."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cem-dir", default="results")
    parser.add_argument("--output-dir", default="results/tianshou_baselines")
    parser.add_argument("--algorithms", default="dqn,ppo,sac")
    parser.add_argument("--households", default="")
    parser.add_argument("--limit-households", type=int, default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--val-len", type=int, default=96)
    parser.add_argument("--eval-len", type=int, default=72)
    parser.add_argument("--train-episode-len", type=int, default=72)
    parser.add_argument("--action-bins", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--strict-torch-determinism", action="store_true")
    parser.add_argument("--log-interval-steps", type=int, default=50_000)
    parser.add_argument("--hidden-sizes", default="64,64")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--dqn-lr", type=float, default=1e-4)
    parser.add_argument("--dqn-train-freq", type=int, default=4)
    parser.add_argument("--dqn-gradient-steps", type=int, default=1)
    parser.add_argument("--dqn-learning-starts", type=int, default=1_000)
    parser.add_argument("--dqn-target-update-freq", type=int, default=10_000)
    parser.add_argument("--dqn-exploration-fraction", type=float, default=0.10)
    parser.add_argument("--dqn-final-eps", type=float, default=0.05)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-rollout-steps", type=int, default=2_048)
    parser.add_argument("--ppo-batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-vf-coef", type=float, default=0.5)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
    parser.add_argument("--sac-lr", type=float, default=3e-4)
    parser.add_argument("--sac-train-freq", type=int, default=8)
    parser.add_argument("--sac-gradient-steps", type=int, default=1)
    parser.add_argument("--sac-learning-starts", type=int, default=5_000)
    parser.add_argument("--sac-alpha", type=float, default=0.2)
    parser.add_argument("--sac-tau", type=float, default=0.005)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.action_bins < 3 or args.action_bins % 2 == 0:
        raise SystemExit("--action-bins must be an odd integer >= 3 so that zero is included.")
    args.gpu_ids = parse_gpu_ids(args.gpu_ids)
    args.hidden_sizes = parse_int_list(args.hidden_sizes)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    args.device = resolve_device(args.device, args.gpu_ids)

    algorithms = [item.lower() for item in parse_csv_list(args.algorithms)]
    unsupported = sorted(set(algorithms) - {"dqn", "ppo", "sac"})
    if unsupported:
        raise SystemExit(f"Unsupported algorithms: {unsupported}")
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
        device=args.device,
        gpu_ids=args.gpu_ids,
        workers=args.workers,
        torch_num_threads=args.torch_num_threads,
        log_interval_steps=args.log_interval_steps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        hidden_sizes=args.hidden_sizes,
        dqn_train_freq=args.dqn_train_freq,
        dqn_gradient_steps=args.dqn_gradient_steps,
        dqn_learning_starts=args.dqn_learning_starts,
        dqn_target_update_freq=args.dqn_target_update_freq,
        dqn_exploration_fraction=args.dqn_exploration_fraction,
        dqn_final_eps=args.dqn_final_eps,
        ppo_rollout_steps=args.ppo_rollout_steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_epochs=args.ppo_epochs,
        sac_train_freq=args.sac_train_freq,
        sac_gradient_steps=args.sac_gradient_steps,
        sac_learning_starts=args.sac_learning_starts,
        sac_alpha=args.sac_alpha,
        sac_tau=args.sac_tau,
        strict_torch_determinism=args.strict_torch_determinism,
    )
    write_reproducibility_files(config, args.output_dir)

    all_rows: list[dict] = []
    for household in households:
        csv_path = os.path.join(args.data_dir, f"{household}.csv")
        cem_path = os.path.join(args.cem_dir, f"{household}_expert.npy")
        print(f"\n>>> Household {household}", flush=True)
        _, eval_df = split_household_df(csv_path, args.val_len)
        deterministic_baselines = {
            "cem": evaluate_cem(cem_path, eval_df, args.eval_len, args.ratio),
            "greedy": evaluate_rule(eval_df, args.eval_len, args.ratio, greedy_policy),
            "company_proxy": evaluate_rule(eval_df, args.eval_len, args.ratio, company_proxy_policy),
            "no_battery": evaluate_rule(eval_df, args.eval_len, args.ratio, no_battery_policy),
        }
        for method, metrics in deterministic_baselines.items():
            metrics["algorithm"] = method
            metrics["library"] = "deterministic"
            metrics["action_space"] = "continuous" if method == "cem" else "rule"
            metrics["action_bins"] = np.nan
            all_rows.append(result_row(household, method, "deterministic", metrics))
            print(f"    {method:<14} {metrics['avg_profit_eur_h']:+.6f} EUR/h", flush=True)

    write_result_tables(all_rows, args.output_dir)
    rl_jobs = make_rl_jobs(args, households, algorithms, seeds)
    run_training_jobs(args, rl_jobs, all_rows)

    results_path, summary_path = write_result_tables(all_rows, args.output_dir)
    summary = pd.read_csv(summary_path)
    print(f"\n>>> Results: {results_path}")
    print(f">>> Summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
