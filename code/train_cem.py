import numpy as np
from tensorboardX import SummaryWriter
from energy_storage_env import EnergyStorageEnv, MIN_SOC, MAX_SOC, BATTERY_CAPACITY_WH, CHARGE_EFFICIENCY, DISCHARGE_EFFICIENCY
import argparse
import os
import pandas as pd

def main():
    # --- 1. 配置参数 ---
    parser = argparse.ArgumentParser(description="CEM Training for Energy Storage - Git Version")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--data_path", type=str, default="../processed_energy_data_germany.csv", help="Path to preprocessed CSV")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size")
    parser.add_argument("--elite_size", type=int, default=10, help="Number of elites")
    parser.add_argument("--n_iter", type=int, default=300, help="Number of iterations")
    parser.add_argument("--alpha", type=float, default=0.8, help="Smoothing factor for update")
    parser.add_argument("--save_dir", type=str, default="policies", help="Directory to save weights")
    parser.add_argument("--res_dir", type=str, default="results", help="Directory to save training history")
    
    args = parser.parse_args()
    
    # 路径准备
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.res_dir, exist_ok=True)
    
    # 固定评估参数
    NUM_EVAL_EPISODES = 20
    FIXED_MONITORING_SEEDS = np.random.RandomState(6202).randint(0, 2**32 - 1, size=NUM_EVAL_EPISODES)
    FIXED_TEST_SEEDS = np.random.RandomState(2026).randint(0, 2**32 - 1, size=50)
    
    np.random.seed(args.seed)

    # --- 2. 环境初始化 ---
    env = EnergyStorageEnv(data_path=args.data_path)
    obs_dim = env.observation_space.shape[0]
    policy_dim = obs_dim + 1  # +1 bias

    # 策略分布初始化
    mu = np.random.uniform(-1, 1, size=policy_dim)
    sigma = np.ones(policy_dim) * 0.5

    writer = SummaryWriter(comment=f"-cem-seed{args.seed}")

    # 历史记录
    history = {
        "iteration": [],
        "cem_reward": [],
        "dumb_baseline": [],
        "self_consumption": [],
        "company_baseline": [],
        "savings_vs_self": []
    }

    # --- 3. 辅助评估函数 ---
    def evaluate(W, seeds):
        total_reward = 0.0
        for s in seeds:
            state, _ = env.reset(seed=int(s))
            episode_reward = 0.0
            while True:
                action_value = np.tanh(np.dot(W[:-1], state) + W[-1])
                action = np.array([action_value * env.action_space.high[0]])
                state, reward, term, trunc, _ = env.step(action)
                episode_reward += reward
                if term or trunc: break
            total_reward += episode_reward
        return total_reward / len(seeds)

    def eval_baselines(seeds):
        # 基准测试汇总
        dumb, self_c, comp = 0, 0, 0
        for s in seeds:
            # Dumb (No action)
            state, _ = env.reset(seed=int(s))
            while True:
                _, r, term, trunc, _ = env.step(np.array([0.0]))
                dumb += r
                if term or trunc: break
            
            # Self-Consumption
            state, info = env.reset(seed=int(s))
            while True:
                net = info['pv_gen'] - info['user_load']
                # 简化逻辑: 有余电就充，缺电就放 (Env 会自动处理 SOC 限制)
                act = np.clip(net, env.action_space.low[0], env.action_space.high[0])
                _, r, term, trunc, info = env.step(np.array([act]))
                self_c += r
                if term or trunc: break
            
            # Company Baseline (Rule based)
            state, info = env.reset(seed=int(s))
            while True:
                mode = info['strategy_mode']
                act = 0
                if mode == 2 or mode == 1: act = env.action_space.high[0] # Negative/Valley
                elif mode == 0: act = env.action_space.low[0] # Peak
                _, r, term, trunc, info = env.step(np.array([act]))
                comp += r
                if term or trunc: break
                
        n = len(seeds)
        return dumb/n, self_c/n, comp/n

    # --- 4. 训练循环 ---
    print(f"Starting CEM training with Seed {args.seed}...")
    for i in range(args.n_iter):
        # 采样种群
        seeds_pop = np.random.randint(0, 2**32-1, size=NUM_EVAL_EPISODES)
        population = np.random.normal(loc=mu, scale=sigma + 1e-7, size=(args.pop_size, policy_dim))
        
        rewards = np.array([evaluate(W, seeds_pop) for W in population])
        
        # 进化更新
        elite_idx = rewards.argsort()[-args.elite_size:]
        elites = population[elite_idx]
        
        mu = args.alpha * mu + (1 - args.alpha) * elites.mean(axis=0)
        sigma = args.alpha * sigma + (1 - args.alpha) * np.maximum(elites.std(axis=0), 0.01)

        # 定期监控 (固定种子)
        d_r, s_r, c_r = eval_baselines(FIXED_MONITORING_SEEDS)
        cem_r = evaluate(mu, FIXED_MONITORING_SEEDS)
        
        # 保存历史
        history["iteration"].append(i)
        history["cem_reward"].append(cem_r)
        history["dumb_baseline"].append(d_r)
        history["self_consumption"].append(s_r)
        history["company_baseline"].append(c_r)
        history["savings_vs_self"].append(cem_r - s_r)

        if i % 10 == 0:
            print(f"Iter {i:3d} | CEM: {cem_r:8.2f} | Self-Cons: {s_r:8.2f} | Company: {c_r:8.2f}")
            writer.add_scalar("Reward/CEM", cem_r, i)

    # --- 5. 保存结果 ---
    # 保存权重
    np.save(os.path.join(args.save_dir, f"weights_seed_{args.seed}.npy"), mu)
    
    # 保存训练历史为 CSV
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(args.res_dir, f"history_seed_{args.seed}.csv"), index=False)
    
    # 最终测试
    final_cem = evaluate(mu, FIXED_TEST_SEEDS)
    print("" + "="*30)
    print(f"Training Complete! Final Test Reward: {final_cem:.2f}")
    print(f"Weights saved to {args.save_dir}")
    print(f"History saved to {args.res_dir}")
    print("="*30)

    writer.close()
    env.close()

if __name__ == "__main__":
    main()
