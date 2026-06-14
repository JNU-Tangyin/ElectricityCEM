# -*- coding: utf-8 -*-
import argparse
import glob
import multiprocessing
import os
import time
import numpy as np
import pandas as pd
from energy_storage_env import EnergyStorageEnv

# ==========================================
# 训练超参数
# ==========================================
N_ITER = 300
POP_SIZE = 150
ELITE_SIZE = 20
ALPHA = 0.85
MIN_SIGMA = 0.05
MAX_W = 1.5
EPISODE_LEN_TRAIN = 72
N_TRAIN_WINDOWS = 3

def train_with_history(args):
    hid, csv_path, save_path, sell_ratio, skip_existing = args
    if skip_existing and os.path.exists(save_path):
        return {"household": hid, "status": "skipped"}
    
    history_path = save_path.replace("_expert.npy", "_history.csv")
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    val_len = 96
    if len(df) < EPISODE_LEN_TRAIN + val_len:
        return {"household": hid, "status": "error", "reason": "data_too_short"}
    
    split_idx = len(df) - val_len
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    val_eval_len = val_len - 24

    env = EnergyStorageEnv(train_df, episode_length_hours=EPISODE_LEN_TRAIN, sell_price_ratio=sell_ratio, action_penalty_coef=0.0)
    env_val = EnergyStorageEnv(val_df, episode_length_hours=val_eval_len, sell_price_ratio=sell_ratio, action_penalty_coef=0.0)
    
    try:
        policy_dim = env.observation_space.shape[0] + 1
        mu = rng.uniform(-0.1, 0.1, size=policy_dim)
        sigma = np.ones(policy_dim) * 0.5

        best_mu = mu.copy()
        best_profit = -np.inf
        history = []

        # ==========================================
        # CEM 核心训练循环
        # 交叉熵方法（CEM）通过不断缩小“优秀参数”的搜索范围来实现进化。
        # 相比于随机搜索，它能更快地向高收益区域靠拢；相比于梯度下降，它不需要计算复杂的导数。
        # ==========================================
        for i in range(N_ITER):
            # 1. 种群采样：从当前的正态分布 (mu, sigma) 中生成 POP_SIZE 个候选策略向量。
            population = rng.normal(loc=mu, scale=sigma, size=(POP_SIZE, policy_dim))
            # 限制权重幅度，防止 tanh 进入饱和区导致策略失效。
            population = np.clip(population, -MAX_W, MAX_W)
            
            rewards = []
            for W in population:
                # 2. 蒙特卡洛评估：
                # 电力环境具有高度的随机性（某天电价极高可能纯属运气），
                # 因此我们选择 N_TRAIN_WINDOWS 个不同的时间起点进行测试并取均值。
                win_profits = []
                for _ in range(N_TRAIN_WINDOWS):
                    t_start = rng.integers(0, max(1, len(train_df) - EPISODE_LEN_TRAIN - 24))
                    state, _ = env.reset(options={"start_idx": t_start})
                    ep_profit = 0.0
                    while True:
                        # 执行线性控制策略
                        action_val = np.tanh(np.dot(W[:-1], state) + W[-1])
                        state, _, done, _, info = env.step(np.array([action_val]))
                        ep_profit += info["profit"]
                        if done: break
                    win_profits.append(ep_profit / EPISODE_LEN_TRAIN)
                rewards.append(np.mean(win_profits))

            # 3. 精英筛选：根据得分排序，选出表现最好的前 ELITE_SIZE 个策略。
            elite_idxs = np.array(rewards).argsort()[-ELITE_SIZE:]
            elites = population[elite_idxs]
            
            # 4. 分布更新：
            # 将下一代的搜索中心 (mu) 移向本代精英的平均值，将搜索半径 (sigma) 缩小。
            # ALPHA 平滑因子确保了进化的稳健性，防止单代异常导致模型异常。
            mu = ALPHA * mu + (1 - ALPHA) * elites.mean(axis=0)
            sigma = ALPHA * sigma + (1 - ALPHA) * np.maximum(elites.std(axis=0), MIN_SIGMA)

            # 5. 跨时段泛化验证：
            # 即便在训练集上表现优异，也可能只是记住了历史。
            # 我们在从未见过的 val_df 上评估当前中心策略 mu 的表现。
            state, _ = env_val.reset(options={"start_idx": 0})
            avg_eval_profit = 0.0
            while True:
                action_val = np.tanh(np.dot(mu[:-1], state) + mu[-1])
                state, _, done, _, info = env_val.step(np.array([action_val]))
                avg_eval_profit += info["profit"]
                if done: break
            avg_eval_profit /= val_eval_len
            
            history.append({'iteration': i, 'mean_reward': np.mean(rewards), 'eval_profit': avg_eval_profit})

            # 只保存泛化表现（验证集收益）最好的那一个版本。
            if avg_eval_profit > best_profit:
                best_profit = avg_eval_profit
                best_mu = mu.copy()

            if (i+1) % 50 == 0:
                print(f"[{hid}] Iter {i+1:3d} | Best Profit: {best_profit:8.4f} EUR/h")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, best_mu)
        pd.DataFrame(history).to_csv(history_path, index=False)
        return {"household": hid, "status": "success", "profit": best_profit}
    
    finally:
        env.close()
        env_val.close()

def main():
    # ==========================================
    # 路径管理与参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="德国家庭储能 CEM 训练脚本")
    parser.add_argument("--data_dir", type=str, required=True, help="预处理后的 CSV 数据文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="训练结果 (expert.npy) 保存目录")
    parser.add_argument("--ratio", type=float, default=1.0, help="卖电/买电价格比例 (sell_price_ratio)")
    parser.add_argument("--skip-existing", action="store_true", help="如果已存在结果则跳过")
    parser.add_argument("--workers", type=int, default=None, help="并行训练的进程数")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    print(f">>> 数据来源目录: {data_dir}")
    print(f">>> 结果保存目录: {output_dir}")

    if not os.path.exists(data_dir):
        print(f"错误：找不到数据目录 {data_dir}，请检查输入。")
        return

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    tasks = [
        (
            os.path.basename(f).replace(".csv", ""), 
            f, 
            os.path.join(output_dir, f"{os.path.basename(f).replace('.csv', '')}_expert.npy"), 
            args.ratio, 
            args.skip_existing
        ) for f in csv_files
    ]

    num_cores = args.workers or max(1, multiprocessing.cpu_count() - 2)
    print(f">>> 训练启动: Ratio={args.ratio} | 进程数={num_cores}")
    
    time_start = time.time()
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(train_with_history, tasks)
    print(f">>> 训练完成，耗时: {(time.time() - time_start)/60:.1f} min.")

if __name__ == "__main__":
    main()
