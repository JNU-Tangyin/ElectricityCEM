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
# 基线与对比逻辑
# ==============================================================================

def run_greedy_backtest(df, length, start_idx, sell_ratio):
    soc, total_profit = 0.5, 0.0
    for step in range(start_idx, start_idx + length):
        row = df.iloc[step]
        soc, port_p = apply_battery_physics(soc, row["pv_gen_wh"] - row["user_load_wh"])
        total_profit += calculate_economics(row["pv_gen_wh"], row["user_load_wh"], port_p, row["price_eur_kwh"], sell_ratio)
    return total_profit / length

def run_proxy_backtest(df, length, start_idx, sell_ratio):
    """
    公司规则Proxy基线(从之前给的'输出策略时间.xlsx'假设推断的)
    逻辑说明：
    1. 负电价模式 (Mode 2): 电价 <= 0.00244 EUR/kWh -> 强制充电
    2. 波峰模式 (Mode 0): 电价处于日内前 25% -> 强制放电
    3. 波谷模式 (Mode 1): 电价处于日内后 25% -> 强制充电
    """
    soc, total_profit = 0.5, 0.0
    df_ranked = df.copy()
    # 计算当日价格排名
    df_ranked["_rank"] = df_ranked.groupby(df_ranked.index.date)["price_eur_kwh"].rank(pct=True)
    
    for step in range(start_idx, start_idx + length):
        row = df_ranked.iloc[step]
        price = row['price_eur_kwh']
        rank = row['_rank']
        
        # 应用解析出的 if-else 规则
        if price <= 0.00244:
            intended_w = MAX_CHARGE_POWER_W
        elif rank >= 0.75:
            intended_w = -MAX_DISCHARGE_POWER_W
        elif rank <= 0.25:
            intended_w = MAX_CHARGE_POWER_W
        else:
            intended_w = 0.0
            
        soc, port_p = apply_battery_physics(soc, intended_w)
        total_profit += calculate_economics(row["pv_gen_wh"], row["user_load_wh"], port_p, price, sell_ratio)
    return total_profit / length

def run_ai_backtest(W, df, length, start_idx, sell_ratio):
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
    parser = argparse.ArgumentParser(description="德国家庭储能 CEM 性能回测脚本")
    parser.add_argument("--data_dir", type=str, required=True, help="CSV 原始数据文件夹路径")
    parser.add_argument("--vectors_dir", type=str, required=True, help="vector (.npy) 所在文件夹路径")
    parser.add_argument("--ratio", type=float, default=1.0, help="卖电/买电价格比例")
    args = parser.parse_args()

    print(f">>> vector搜索目录: {args.vectors_dir}")
    print(f">>> 数据来源目录: {args.data_dir}")

    if not os.path.exists(args.vectors_dir):
        print(f"错误：找不到vector目录 {args.vectors_dir}")
        return

    vector_files = sorted(glob.glob(os.path.join(args.vectors_dir, "*_expert.npy")))
    
    if not vector_files:
        print(f"警告：在 {args.vectors_dir} 中未找到任何vector文件。")
        return

    print(f">>> 正在对 {len(vector_files)} 个家庭进行多基线性能审计 (使用解析版公司规则)...")
    
    results = []
    for v_path in vector_files:
        hid = os.path.basename(v_path).replace("_expert.npy", "")
        csv_path = os.path.join(args.data_dir, f"{hid}.csv")
        
        if not os.path.exists(csv_path):
            continue
        
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        length = len(df) - 24 
        W = np.load(v_path)
        
        gr_p = run_greedy_backtest(df, length, 0, args.ratio)
        pr_p = run_proxy_backtest(df, length, 0, args.ratio)
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

    df_res = pd.DataFrame(results)
    summary_path = os.path.join(args.vectors_dir, "backtest_summary_handover.csv")
    df_res.to_csv(summary_path, index=False)
    
    print(f"\n>>> 性能审计完成！")
    print(f">>> CEM 平均胜率 (vs Proxy): {(df_res['CEM_minus_proxy'] > 0).mean():.1%}")
    print(f">>> 详细报表已保存至: {summary_path}")

if __name__ == "__main__":
    main()
