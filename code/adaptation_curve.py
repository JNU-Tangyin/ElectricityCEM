import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from energy_storage_env import EnergyStorageEnv

# 统一评估规模，确保统计分布一致
EVAL_EPISODES = 15 

def get_no_battery_baseline(data_path, n_episodes=EVAL_EPISODES):
    temp_env = EnergyStorageEnv(data_path=data_path)
    rewards = []
    for _ in range(n_episodes):
        state, _ = temp_env.reset()
        ep_reward = 0
        while True:
            # 物理上的“无电池”行为
            state, reward, term, trunc, _ = temp_env.step(np.array([0.0]))
            ep_reward += reward
            if term or trunc: break
        rewards.append(ep_reward)
    temp_env.close()
    return np.mean(rewards)

def evaluate_policy(env, W, n_episodes=EVAL_EPISODES):
    """
    对策略进行稳健评估。
    """
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0
        while True:
            # 策略推理
            action_val = np.tanh(np.dot(W[:-1], state) + W[-1])
            scaled_action = action_val * env.action_space.high[0]
            action = np.clip(np.array([scaled_action]), env.action_space.low, env.action_space.high)
            
            state, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc: break
        rewards.append(ep_reward)
    return np.mean(rewards)

def run_adaptation_sim(data_path, base_reward, base_weights_path=None, n_iter=100, pop_size=50):
    """
    模拟适配过程。
    """
    # 每次运行创建全新的环境实例，彻底杜绝状态污染 (Fix #3)
    env = EnergyStorageEnv(data_path=data_path)
    policy_dim = env.observation_space.shape[0] + 1
    
    # 1. 初始化
    if base_weights_path and os.path.exists(base_weights_path):
        print(f"Starting from BASE vector: {os.path.basename(base_weights_path)}")
        loaded_weights = np.load(base_weights_path)
        
        # 检查维度是否匹配
        if loaded_weights.shape[0] != policy_dim:
            raise ValueError(f"Dimension Mismatch! Environment needs {policy_dim}, but loaded {loaded_weights.shape[0]}.")
            
        mu = loaded_weights
        sigma = np.ones(policy_dim) * 0.2
    else:
        print("Starting FROM SCRATCH (Random Initialization)")
        mu = np.random.randn(policy_dim) * 0.1
        sigma = np.ones(policy_dim) * 0.5
    
    history = []
    # Fix #2: 确保精英数量足够稳定更新
    elite_count = max(10, int(pop_size * 0.2)) 
    
    # 2. 适配循环
    for i in range(n_iter + 1):
        if i == 0:
            reward = evaluate_policy(env, mu, n_episodes=EVAL_EPISODES)
        else:
            population = np.random.normal(loc=mu, scale=sigma + 1e-7, size=(pop_size, policy_dim))
            pop_rewards = np.array([evaluate_policy(env, W, n_episodes=EVAL_EPISODES) for W in population])
            
            # CEM 更新逻辑
            elite_idx = pop_rewards.argsort()[-elite_count:]
            elites = population[elite_idx]
            mu = 0.8 * mu + 0.2 * elites.mean(axis=0)
            sigma = 0.8 * sigma + 0.2 * np.maximum(elites.std(axis=0), 0.01)
            
            reward = evaluate_policy(env, mu, n_episodes=EVAL_EPISODES)
        
        # 数值稳定性保护：防止 base_reward 接近 0 导致的计算爆炸
        denom = abs(base_reward) if abs(base_reward) > 1e-6 else 1.0
        imp_pct = (reward - base_reward) / denom * 100
        history.append(imp_pct)
        
        if i % 20 == 0:
            print(f" Iter {i:2d} | Improvement: {imp_pct:6.2f}%")
            
    env.close()
    return history

def main():
    np.random.seed(42) 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    # 修正：使用纯德国训练的权重作为适配的“底座”
    weights_path = os.path.join(base_dir, "policies/weights_seed_42.npy")
    france_data = os.path.join(base_dir, ".../processed_energy_data_france.csv")
    
    # 1. 首先独立计算 Baseline
    base_reward = get_no_battery_baseline(france_data)
    
    print("\n[Scenario 1: Transfer Learning from German Base]")
    transfer_history = run_adaptation_sim(france_data, base_reward, base_weights_path=weights_path, n_iter=100)
    
    print("\n[Scenario 2: From Scratch]")
    scratch_history = run_adaptation_sim(france_data, base_reward, base_weights_path=None, n_iter=100)
    
    # --- 新增：保存数据到 CSV ---
    results_df = pd.DataFrame({
        'Iteration': np.arange(len(transfer_history)),
        'Transfer_Improvement_Pct': transfer_history,
        'Scratch_Improvement_Pct': scratch_history
    })
    csv_save_path = os.path.join(base_dir, "results/adaptation_comparison_results.csv")
    results_df.to_csv(csv_save_path, index=False)
    print(f"Data saved to: {csv_save_path}")
    
    # 绘图
    plt.figure(figsize=(12, 7))
    plt.plot(transfer_history, label='Adaptation from Universal Base', color='#1f77b4', linewidth=2.5)
    plt.plot(scratch_history, label='Training from Scratch', color='#7f7f7f', linestyle='--', linewidth=2)
    
    gap = transfer_history[0] - scratch_history[0]
    plt.annotate(f'Initial Advantage: +{gap:.1f}%', 
                 xy=(0, transfer_history[0]), 
                 xytext=(15, transfer_history[0]+5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    
    plt.title("The 'Zero-shot' Benefit: Universal Base vs. Cold Start (France)", fontsize=14)
    plt.xlabel("Iterations (Days of Operational Feedback)", fontsize=12)
    plt.ylabel("Improvement over No-Battery Baseline (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    save_path = os.path.join(base_dir, "plots/adaptation_final6.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSimulation Finalized. Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
