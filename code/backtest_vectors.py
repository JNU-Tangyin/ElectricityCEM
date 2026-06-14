# -*- coding: utf-8 -*-
import argparse
import glob
import os
import json
import numpy as np
import pandas as pd
from energy_storage_env import (
    MAX_CHARGE_POWER_W,
    MAX_DISCHARGE_POWER_W,
    EnergyStorageEnv,
    apply_battery_physics,
    calculate_economics,
)

# ==============================================================================
# 基线与对比逻辑 (Baseline Comparison Logic)
# 为了证明 AI 策略的有效性，我们需要将其与三种基线进行对比：
# 1. Greedy: 简单的自发自用逻辑。
# 2. Proxy: 模拟公司现有的规则调度策略。
# 3. No-Battery (隐性对比): 即 c_im 的原始数值。
# ==============================================================================

def expand_company_schedule(strategy_path):
    """
    解析公司提供的加密/压缩格式的时间段策略。
    """
    if not strategy_path or not os.path.exists(strategy_path): return pd.Series(dtype=float)
    df = pd.read_csv(strategy_path, encoding="utf-8-sig")
    rows = []
    for _, row in df.iterrows():
        try:
            raw = json.loads(row["时间段"].replace("'", '"'))
            day = pd.to_datetime(raw.get("dataTime") or row["日期"]).normalize()
            for slot in raw["times"]:
                rows.append((day + pd.Timedelta(hours=int(slot["startTime"].split(":")[0])), int(slot["mode"])))
        except: continue
    if not rows: return pd.Series(dtype=float)
    return pd.DataFrame(rows, columns=["timestamp", "mode"]).drop_duplicates("timestamp").set_index("timestamp").sort_index()["mode"]

def run_greedy_backtest(df, length, start_idx, sell_ratio):
    """
    自发自用优先(Greedy)基线(电池仅凭物理本能运行时的收益)。
    """
    soc, total_profit = 0.5, 0.0
    for step in range(start_idx, start_idx + length):
        row = df.iloc[step]
        # 贪婪逻辑：有多少余电充多少，有多少负载放多少。
        soc, port_p = apply_battery_physics(soc, row["pv_gen_wh"] - row["user_load_wh"])
        total_profit += calculate_economics(row["pv_gen_wh"], row["user_load_wh"], port_p, row["price_eur_kwh"], sell_ratio)
    return total_profit / length

def run_proxy_backtest(df, length, start_idx, sell_ratio, direct_modes):
    """
    公司规则代理(Proxy)基线。
    理由：模拟公司目前的规则调度器（基于历史表或价格分位数）。
    """
    soc, total_profit = 0.5, 0.0
    df_ranked = df.copy()
    # 计算价格分位数，用于模拟低吸高抛规则。
    df_ranked["_rank"] = df_ranked.groupby(df_ranked.index.date)["price_eur_kwh"].rank(pct=True)
    
    for step in range(start_idx, start_idx + length):
        row = df_ranked.iloc[step]
        ts = row.name
        # 优先使用历史策略表，无记录则使用分位数规则。
        if ts in direct_modes.index:
            mode = direct_modes.loc[ts]
            intended_w = MAX_CHARGE_POWER_W if mode in [1,2] else (-MAX_DISCHARGE_POWER_W if mode==0 else 0.0)
        else:
            intended_w = MAX_CHARGE_POWER_W if row["_rank"] <= 0.25 else (-MAX_DISCHARGE_POWER_W if row["_rank"] >= 0.75 else 0.0)
        
        soc, port_p = apply_battery_physics(soc, intended_w)
        total_profit += calculate_economics(row["pv_gen_wh"], row["user_load_wh"], port_p, row["price_eur_kwh"], sell_ratio)
    return total_profit / length

def run_ai_backtest(W, df, length, start_idx, sell_ratio):
    """
    CEM vector回测。
    理由：使用训练好的线性权重进行推理，验证其在完整历史时段内的盈利能力。
    """
    env = EnergyStorageEnv(df, episode_length_hours=length, sell_price_ratio=sell_ratio)
    state, _ = env.reset(options={"start_idx": start_idx})
    total_profit = 0.0
    for _ in range(length):
        action_val = np.tanh(np.dot(W[:-1], state) + W[-1])
        state, _, done, _, info = env.step(np.array([action_val]))
        total_profit += info["profit"]
        if done: break
    env.close()
    return total_profit / length

def main():
    # ==============================================================================
    # 路径管理与参数解析
    # ==============================================================================
    parser = argparse.ArgumentParser(description="德国家庭储能 CEM 性能回测脚本")
    parser.add_argument("--data_dir", type=str, required=True, help="CSV 原始数据文件夹路径")
    parser.add_argument("--vectors_dir", type=str, required=True, help="专家vector (.npy) 所在文件夹路径")
    parser.add_argument("--strategy_path", type=str, default=None, help="公司历史策略时间表路径 (可选，用于 Proxy 对比)")
    parser.add_argument("--ratio", type=float, default=1.0, help="卖电/买电价格比例")
    args = parser.parse_args()

    print(f">>> vector搜索目录: {args.vectors_dir}")
    print(f">>> 数据来源目录: {args.data_dir}")

    if not os.path.exists(args.vectors_dir):
        print(f"错误：找不到vector目录 {args.vectors_dir}")
        return

    # 加载公司基准策略。
    direct_modes = expand_company_schedule(args.strategy_path)
    vector_files = sorted(glob.glob(os.path.join(args.vectors_dir, "*_expert.npy")))
    
    if not vector_files:
        print(f"警告：在 {args.vectors_dir} 中未找到任何专家vector文件。")
        return

    print(f">>> 正在对 {len(vector_files)} 个家庭进行多基线性能审计...")
    
    results = []
    for v_path in vector_files:
        hid = os.path.basename(v_path).replace("_expert.npy", "")
        csv_path = os.path.join(args.data_dir, f"{hid}.csv")
        
        if not os.path.exists(csv_path):
            print(f"跳过 HID {hid}: 找不到对应的 CSV 数据。")
            continue
        
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        length = len(df) - 24 
        W = np.load(v_path)
        
        # 运行对比实验。
        gr_p = run_greedy_backtest(df, length, 0, args.ratio)
        pr_p = run_proxy_backtest(df, length, 0, args.ratio, direct_modes)
        ai_p = run_ai_backtest(W, df, length, 0, args.ratio)
        
        results.append({
            "household": hid,
            "ai_profit": ai_p,
            "greedy_profit": gr_p,
            "proxy_profit": pr_p,
            "ai_minus_greedy": ai_p - gr_p,
            "ai_minus_proxy": ai_p - pr_p
        })
        print(f"HID {hid:<8} | AI vs Greedy: {ai_p-gr_p:+.5f} | AI vs Proxy: {ai_p-pr_p:+.5f}")

    # 生成汇总报告。
    df_res = pd.DataFrame(results)
    summary_path = os.path.join(args.vectors_dir, "backtest_summary_handover.csv")
    df_res.to_csv(summary_path, index=False)
    
    print(f"\n>>> 回测完成。")
    print(f">>> CEM 平均胜率 (vs Greedy): {(df_res['ai_minus_greedy'] > 0).mean():.1%}")
    print(f">>> 详细报表已保存至: {summary_path}")

if __name__ == "__main__":
    main()
