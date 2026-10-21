# -*- coding: utf-8 -*-
"""
CEM 训练，与 train_cem.py 的区别是同一代内所有候选共用同一批训练窗口。
新增逐代的 mu/sigma 轨迹、单户计时与 checkpoint 代数记录，用于诊断收敛；
超参数、validation 逐代挑 checkpoint、val_len=96 均保持一致。

"""
import argparse
import glob
import multiprocessing
import os
import time
import numpy as np
import pandas as pd

# 目录结构固定为 code/(本脚本) 与同级的 data/、results/，由脚本自身位置回推默认值，
# 保证换一台机器、换一个安装路径都能直接运行，不依赖任何人的本地绝对路径。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BASE = os.path.dirname(_SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(_DEFAULT_BASE, "data")
DEFAULT_OUTPUT_DIR = os.path.join(_DEFAULT_BASE, "results_better")
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
    hid, csv_path, save_path, sell_ratio, skip_existing, seed_seq = args
    # 逐户计时：整批耗时除以户数会被并行度掩盖，论文里要报的是单户训练成本。
    t0 = time.time()
    if skip_existing and os.path.exists(save_path):
        return {"household": hid, "status": "skipped"}

    history_path = save_path.replace("_expert.npy", "_history.csv")
    # seed_seq 是从全局 --seed 派生出的该户专属子种子（见 main() 里的 SeedSequence.spawn），
    # 保证同一个全局种子重跑整批训练能得到完全一致的结果，同时各户之间的随机数互不相关。
    rng = np.random.default_rng(seed_seq)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    val_len = 96
    # +1：训练窗口起点采样需要 len(train_df) > EPISODE_LEN_TRAIN 才有余量（见下面 -1 的注释），
    # 否则 len(df) 恰好等于 EPISODE_LEN_TRAIN+val_len 时 train_df 长度会正好等于 EPISODE_LEN_TRAIN，
    # 训练循环最后一步会在 step() 里越界（IndexError）。
    if len(df) < EPISODE_LEN_TRAIN + val_len + 1:
        return {"household": hid, "status": "error", "reason": "data_too_short"}
    
    split_idx = len(df) - val_len
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    # -1 是必须的：EnergyStorageEnv.step() 即便在最后一步 (done=True) 也会用
    # 递增后的 current_step 再调用一次 _get_obs()，所以 episode 长度必须比
    # 可用行数少 1，否则最后一步会越界（IndexError）。
    val_eval_len = val_len - 1

    env = EnergyStorageEnv(train_df, episode_length_hours=EPISODE_LEN_TRAIN, sell_price_ratio=sell_ratio, action_penalty_coef=0.0)
    env_val = EnergyStorageEnv(val_df, episode_length_hours=val_eval_len, sell_price_ratio=sell_ratio, action_penalty_coef=0.0)
    
    try:
        policy_dim = env.observation_space.shape[0] + 1
        mu = rng.uniform(-0.1, 0.1, size=policy_dim)
        sigma = np.ones(policy_dim) * 0.5

        best_mu = mu.copy()
        best_profit = -np.inf
        best_iter = -1
        history = []

        # 逐代留下搜索分布本身，而不只是它的得分。
        # 只存最终的 best_mu 就无法回答「参数是真的收敛了，还是奖励先平了、参数还在漂」，
        # 而这个问题事后补不了——没有轨迹就只能重跑一遍训练。
        traj_mu = np.empty((N_ITER, policy_dim))
        traj_sigma = np.empty((N_ITER, policy_dim))

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
            
            # 2. 蒙特卡洛评估的窗口：本代所有候选共用同一批。
            # 若每个候选各抽各的，两个候选的分数差里就混入了「谁抽到的窗口更好赚」，
            # 而窗口之间的收益差异远大于候选之间的差异，精英筛选会主要在选运气。
            # 共用之后比较变成配对的，窗口难易被差分掉。换代时重抽，避免 mu 记住固定几段行情。
            # max(0, ...) 是最大允许起点（含），rng.integers 的 high 是排他的，所以要 +1；
            # 其中的 -1 原因同 val_eval_len 那处注释。
            win_starts = rng.integers(
                0, max(0, len(train_df) - EPISODE_LEN_TRAIN - 1) + 1, size=N_TRAIN_WINDOWS)

            rewards = []
            for W in population:
                win_profits = []
                for t_start in win_starts:
                    state, _ = env.reset(options={"start_idx": int(t_start)})
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

            # 记在更新之后、验证之前，使第 i 行的 mu 正好是下面被验证出 eval_profit 的那一个，
            # 轨迹与 history 逐行对齐。
            traj_mu[i] = mu
            traj_sigma[i] = sigma

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
                best_iter = i

            if (i+1) % 50 == 0:
                print(f"[{hid}] Iter {i+1:3d} | Best Profit: {best_profit:8.4f} EUR/h")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, best_mu)
        np.save(save_path.replace("_expert.npy", "_traj_mu.npy"), traj_mu)
        np.save(save_path.replace("_expert.npy", "_traj_sigma.npy"), traj_sigma)
        pd.DataFrame(history).to_csv(history_path, index=False)
        # best_iter 指出 checkpoint 取自第几代。同一户换种子重训时它差得很远，
        # 是解释参数不可复现的线索之一，所以要留下来而不是只留 best_mu。
        return {"household": hid, "status": "success", "profit": best_profit,
                "best_iter": best_iter, "seconds": time.time() - t0}
    
    finally:
        env.close()
        env_val.close()

def main():
    # ==========================================
    # 路径管理与参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="德国家庭储能 CEM 训练脚本")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="预处理后的 CSV 数据文件夹路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="训练结果 (expert.npy) 保存目录")
    parser.add_argument("--ratio", type=float, default=1.0, help="卖电/买电价格比例 (sell_price_ratio)")
    parser.add_argument("--skip-existing", action="store_true", help="如果已存在结果则跳过")
    parser.add_argument("--workers", type=int, default=None, help="并行训练的进程数")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子，用于派生每户互相独立的训练种子（保证整批训练可复现）")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    print(f">>> 数据来源目录: {data_dir}")
    print(f">>> 结果保存目录: {output_dir}")

    if not os.path.exists(data_dir):
        print(f"错误：找不到数据目录 {data_dir}，请检查输入。")
        return

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # 每户从同一个全局种子派生出各自独立的子种子：SeedSequence.spawn 保证子种子之间
    # 统计独立（不会像多个进程共用同一个种子那样产生跨户关联），且只要 --seed 和
    # csv_files 的顺序不变，重跑整批训练能得到逐户完全一致的结果。
    household_seeds = np.random.SeedSequence(args.seed).spawn(len(csv_files))

    tasks = [
        (
            os.path.basename(f).replace(".csv", ""),
            f,
            os.path.join(output_dir, f"{os.path.basename(f).replace('.csv', '')}_expert.npy"),
            args.ratio,
            args.skip_existing,
            household_seeds[i],
        ) for i, f in enumerate(csv_files)
    ]

    num_cores = args.workers or max(1, multiprocessing.cpu_count() - 2)
    print(f">>> 训练启动: Ratio={args.ratio} | 全局种子={args.seed} | 进程数={num_cores}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_run_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"global_seed={args.seed}\n")
        f.write(f"ratio={args.ratio}\n")
        f.write(f"n_households={len(csv_files)}\n")
        f.write(f"data_dir={data_dir}\n")
        f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    time_start = time.time()
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(train_with_history, tasks)
    print(f">>> 训练完成，耗时: {(time.time() - time_start)/60:.1f} min.")

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(output_dir, "training_summary.csv"), index=False)
    ok = summary[summary.status == "success"]
    if not ok.empty:
        print(f">>> 成功 {len(ok)}/{len(summary)} 户 | "
              f"单户耗时中位 {ok.seconds.median()/60:.1f} min（{num_cores} 进程并行下的墙钟）| "
              f"checkpoint 代数中位 {ok.best_iter.median():.0f}/{N_ITER}")

if __name__ == "__main__":
    main()
