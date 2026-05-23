# -*- coding: utf-8 -*-
import argparse
import glob
import multiprocessing
import os
import time
import numpy as np
import pandas as pd
from energy_storage_env import EnergyStorageEnv

N_ITER = 300
POP_SIZE = 100
ELITE_SIZE = 10
ALPHA = 0.8
MIN_SIGMA = 1e-2
EPISODE_LEN = 24 


def train_single_household(args):
    (
        hid,
        csv_path,
        save_path,
        sell_ratio,
        episode_len,
        n_iter,
        pop_size,
        elite_size,
        alpha,
    ) = args

    np.random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    env = EnergyStorageEnv(
        df,
        episode_length_hours=episode_len,
        sell_price_ratio=sell_ratio,
        action_penalty_coef=0.0,
    )
    policy_dim = env.observation_space.shape[0] + 1

    mu = np.random.uniform(-1.0, 1.0, size=policy_dim)
    sigma = np.ones(policy_dim) * 0.5

    best_mu = mu.copy()
    best_reward = -np.inf
    max_start = max(0, len(df) - episode_len - 24)

    for _ in range(n_iter):
        population = np.random.normal(loc=mu, scale=sigma, size=(pop_size, policy_dim))
        train_start = np.random.randint(0, max_start) if max_start > 0 else 0

        rewards = []
        for W in population:
            state, _ = env.reset(options={"start_idx": train_start})
            ep_reward = 0.0
            for _ in range(episode_len):
                action_val = np.tanh(np.dot(W[:-1], state) + W[-1])
                state, _, done, _, info = env.step(np.array([action_val]))
                ep_reward += info["profit"]
                if done:
                    break
            rewards.append(ep_reward / episode_len)

        elite_idxs = np.array(rewards).argsort()[-elite_size:]
        elites = population[elite_idxs]
        mu = alpha * mu + (1 - alpha) * elites.mean(axis=0)
        sigma = alpha * sigma + (1 - alpha) * np.maximum(elites.std(axis=0), MIN_SIGMA)

        state, _ = env.reset(options={"start_idx": 0})
        eval_reward = 0.0
        for _ in range(episode_len):
            action_val = np.tanh(np.dot(mu[:-1], state) + mu[-1])
            state, _, done, _, info = env.step(np.array([action_val]))
            eval_reward += info["profit"]
            if done:
                break
        eval_reward /= episode_len

        if eval_reward > best_reward:
            best_reward = eval_reward
            best_mu = mu.copy()

    env.close()
    np.save(save_path, best_mu)
    return {
        "household": hid,
        "best_reward": best_reward,
        "episode_len": episode_len,
        "sell_ratio": sell_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Train per-household expert vectors.")
    parser.add_argument("--ratio", type=float, required=True, help="Sell/buy price ratio.")
    parser.add_argument(
        "--episode-len",
        type=int,
        default=EPISODE_LEN,
        help="CEM episode length in hours (24 ~= old smooth demo; 168 = aggressive arbitrage).",
    )
    parser.add_argument("--n-iter", type=int, default=N_ITER)
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--elite-size", type=int, default=ELITE_SIZE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(base_dir, "00.data/preprocessed/german_households_weather")
    output_dir = os.path.join(base_dir, f"04.results/expert_vectors_71_ratio_{args.ratio}")

    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    tasks = [
        (
            os.path.basename(f).replace(".csv", ""),
            f,
            os.path.join(output_dir, f"{os.path.basename(f).replace('.csv', '')}_expert.npy"),
            args.ratio,
            args.episode_len,
            args.n_iter,
            args.pop_size,
            args.elite_size,
            args.alpha,
        )
        for f in csv_files
    ]

    print(
        f">>> Training ratio={args.ratio}, episode={args.episode_len}h, "
        f"iter={args.n_iter}, pop={args.pop_size} -> {output_dir}"
    )
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(train_single_household, tasks)

    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "evolution_summary.csv"), index=False
    )
    print(">>> Done.")


if __name__ == "__main__":
    main()
