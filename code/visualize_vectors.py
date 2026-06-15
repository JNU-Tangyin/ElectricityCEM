# -*- coding: utf-8 -*-
import argparse
import glob
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from energy_storage_env import (
    EnergyStorageEnv, 
    MAX_CHARGE_POWER_W, 
    MAX_DISCHARGE_POWER_W,
    apply_battery_physics,
    calculate_economics
)


def calculate_validation_baselines(data_path, ratio):
    #基于'输出策略时间.xlsx'假设的公司if-else逻辑计算验证集基准。
    if not os.path.exists(data_path): return 0.0, 0.0, 0.0
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    val_len = 96
    val_df = df.iloc[-val_len:].copy()
    eval_len = val_len - 24 
    
    val_df["_rank"] = val_df.groupby(val_df.index.date)["price_eur_kwh"].rank(pct=True)

    g_profit, n_profit, p_profit = 0.0, 0.0, 0.0
    g_soc, p_soc = 0.5, 0.5
    
    for i in range(eval_len):
        row = val_df.iloc[i]
        price = row['price_eur_kwh']
        rank = row['_rank']
        pv, load = row['pv_gen_wh'], row['user_load_wh']

        # Greedy
        g_soc, g_port = apply_battery_physics(g_soc, pv - load)
        g_profit += calculate_economics(pv, load, g_port, price, ratio)
        
        # No-Battery
        n_profit += calculate_economics(pv, load, 0.0, price, ratio)

        # Company Proxy
        if price <= 0.00244:
            p_intent = MAX_CHARGE_POWER_W
        elif rank >= 0.75:
            p_intent = -MAX_DISCHARGE_POWER_W
        elif rank <= 0.25:
            p_intent = MAX_CHARGE_POWER_W
        else:
            p_intent = 0.0
        p_soc, p_port = apply_battery_physics(p_soc, p_intent)
        p_profit += calculate_economics(pv, load, p_port, price, ratio)
        
    return (g_profit / eval_len), (n_profit / eval_len), (p_profit / eval_len)

def plot_training_curve(hid, history_path, data_path, output_dir, ratio):
    #绘制训练进化曲线。
    if not os.path.exists(history_path): return
    history_df = pd.read_csv(history_path)
    greedy_avg, nobatt_avg, proxy_avg = calculate_validation_baselines(data_path, ratio)
    
    plt.figure(figsize=(12, 7), dpi=200)
    iterations = history_df['iteration']
    
    plt.plot(iterations, history_df['mean_reward'], color='blue', alpha=0.8, label='Population Mean Reward (Training)')
    
    plt.axhline(y=greedy_avg, color='orange', linestyle='--', label='Baseline: Greedy', alpha=0.7)
    plt.axhline(y=nobatt_avg, color='red', linestyle='--', label='Baseline: No Battery', alpha=0.7)
    
    plt.title(f"CEM Evolution Progress: Household {hid}", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration")
    plt.ylabel("Average Profit (EUR/h)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right')
    
    save_path = os.path.join(output_dir, f"training_curve_{hid}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

def plot_strategy_behavior(hid, data_path, weights_path, output_dir, ratio):
    #绘制3天的行为审计图
    if not os.path.exists(data_path) or not os.path.exists(weights_path): return
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df["_rank"] = df.groupby(df.index.date)["price_eur_kwh"].rank(pct=True)
    W = np.load(weights_path)
    
    test_len = 72
    view_df = df.iloc[:test_len].copy()
    
    env = EnergyStorageEnv(view_df, episode_length_hours=test_len, sell_price_ratio=ratio)
    state, _ = env.reset(options={'start_idx': 0})
    
    results = []
    ai_soc, greedy_soc, proxy_soc = 0.5, 0.5, 0.5
    c_ai, c_gr, c_pr, c_no = 0.0, 0.0, 0.0, 0.0

    for i in range(test_len):
        row = view_df.iloc[i]
        price = row['price_eur_kwh']
        rank = row['_rank']
        pv, load = row['pv_gen_wh'], row['user_load_wh']

        # CEM
        action_val = np.tanh(np.dot(W[:-1], state) + W[-1])
        ai_soc, ai_port = apply_battery_physics(ai_soc, action_val * MAX_CHARGE_POWER_W)
        p_ai = calculate_economics(pv, load, ai_port, price, ratio)
        
        # Greedy
        greedy_soc, gr_port = apply_battery_physics(greedy_soc, pv - load)
        p_gr = calculate_economics(pv, load, gr_port, price, ratio)

        # Proxy (解析版逻辑)
        if price <= 0.00244:
            p_intent = MAX_CHARGE_POWER_W
        elif rank >= 0.75:
            p_intent = -MAX_DISCHARGE_POWER_W
        elif rank <= 0.25:
            p_intent = MAX_CHARGE_POWER_W
        else:
            p_intent = 0.0
        proxy_soc, pr_port = apply_battery_physics(proxy_soc, p_intent)
        p_pr = calculate_economics(pv, load, pr_port, price, ratio)

        # No Battery
        p_no = calculate_economics(pv, load, 0.0, price, ratio)

        c_ai += p_ai; c_gr += p_gr; c_pr += p_pr; c_no += p_no

        results.append({
            'time': view_df.index[i], 'price': price,
            'ai_soc': ai_soc, 'greedy_soc': greedy_soc, 'proxy_soc': proxy_soc,
            'ai_port': ai_port, 'ai_adv': c_ai - c_no, 'gr_adv': c_gr - c_no, 'pr_adv': c_pr - c_no
        })
        if i < test_len - 1: state, _, _, _, _ = env.step(np.array([action_val]))
    
    env.close()
    res_df = pd.DataFrame(results)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, dpi=200)
    
    # Subplot 1: SOC
    ax1.set_title(f'Strategy Behavior: SOC Comparison (HID {hid})', fontsize=14, fontweight='bold')
    ax1_price = ax1.twinx()
    p_line, = ax1_price.plot(res_df['time'], res_df['price'], color='red', alpha=0.2, linestyle='--', label='Market Price')
    ax1_price.set_ylabel('Price (EUR/kWh)', color='red')
    
    g_line, = ax1.plot(res_df['time'], res_df['greedy_soc'], color='gray', label='Greedy SOC', alpha=0.4)
    x_line, = ax1.plot(res_df['time'], res_df['proxy_soc'], color='green', label='Proxy SOC (Inferred)', alpha=0.4, linestyle=':')
    a_line, = ax1.plot(res_df['time'], res_df['ai_soc'], color='blue', label='AI Optimized SOC', linewidth=2.5)
    ax1.set_ylabel('Battery SOC'); ax1.set_ylim(0, 1.1)
    
    lines = [a_line, x_line, g_line, p_line]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')

    # Subplot 2: Power
    ax2.set_title('AI Charging Power (W)', fontsize=14, fontweight='bold')
    ax2.bar(res_df['time'], res_df['ai_port'], color=['#1f77b4' if p > 0 else '#ff7f0e' for p in res_df['ai_port']], alpha=0.6, width=0.03)
    ax2.axhline(0, color='black', lw=1); ax2.set_ylabel('Power (W)')

    # Subplot 3: Advantage
    ax3.set_title('Financial Comparison: Advantage vs No-Battery', fontsize=14, fontweight='bold')
    ax3.axhline(0, color='black', lw=2, label='Baseline: No Battery')
    ax3.plot(res_df['time'], res_df['gr_adv'], color='gray', linestyle='--', label='Greedy Advantage', alpha=0.6)
    ax3.plot(res_df['time'], res_df['pr_adv'], color='green', linestyle=':', label='Proxy Advantage', alpha=0.6)
    ax3.plot(res_df['time'], res_df['ai_adv'], color='orange', linewidth=3, label='AI Strategy (Optimized)')
    ax3.fill_between(res_df['time'], res_df['ai_adv'], res_df['pr_adv'], color='orange', alpha=0.1, label='AI Surplus vs Proxy')
    ax3.set_ylabel('Cumulative Advantage (EUR)'); ax3.grid(True, linestyle='--', alpha=0.3); ax3.legend(loc='upper left')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.tight_layout()
    save_path = os.path.join(output_dir, f"strategy_analysis_{hid}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path

def main():
    parser = argparse.ArgumentParser(description="德国家庭储能 AI 可视化工具 - 交付版")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--vectors_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hids", type=str, default="11098,10169,10401")
    parser.add_argument("--ratio", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    for hid in args.hids.split(","):
        hid = hid.strip()
        print(f">>> 正在为 HID {hid} 生成多基线审计图...")
        history_path = os.path.join(args.vectors_dir, f"{hid}_history.csv")
        data_path = os.path.join(args.data_dir, f"{hid}.csv")
        weights_path = os.path.join(args.vectors_dir, f"{hid}_expert.npy")
        
        p1 = plot_training_curve(hid, history_path, data_path, args.output_dir, args.ratio)
        p2 = plot_strategy_behavior(hid, data_path, weights_path, args.output_dir, args.ratio)
        if p1 and p2: print(f"    - 审计图组已完成。")

if __name__ == "__main__":
    main()
