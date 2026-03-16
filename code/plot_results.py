import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_training_history(res_dir, seed, plots_dir):
    file_path = os.path.join(res_dir, f"history_seed_{seed}.csv")
    if not os.path.exists(file_path):
        print(f"Error: Could not find results for seed {seed} at {file_path}")
        return

    df = pd.read_csv(file_path)
    os.makedirs(plots_dir, exist_ok=True)

    # --- Plot 1: Reward Curves ---
    plt.figure(figsize=(12, 6))
    plt.plot(df['iteration'], df['cem_reward'], label='CEM Optimized Policy', linewidth=2)
    plt.plot(df['iteration'], df['self_consumption'], '--', label='Self-Consumption Baseline', alpha=0.8)
    plt.plot(df['iteration'], df['company_baseline'], ':', label='Company Rule-based Policy', alpha=0.8)
    
    plt.xlabel('Iteration')
    plt.ylabel('Reward (Net Benefit/Cost)')
    plt.title(f'CEM Training Progress (Seed {seed})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path_1 = os.path.join(plots_dir, f"training_curves_seed_{seed}.png")
    plt.savefig(save_path_1, dpi=300)
    print(f"Saved training curves to: {save_path_1}")

    # --- Plot 2: Savings Over Time ---
    plt.figure(figsize=(12, 6))
    plt.fill_between(df['iteration'], df['savings_vs_self'], color='green', alpha=0.2, label='Extra Savings')
    plt.plot(df['iteration'], df['savings_vs_self'], color='darkgreen', label='Savings vs Self-Consumption')
    
    plt.xlabel('Iteration')
    plt.ylabel('Additional Savings (Currency)')
    plt.title('Incremental Improvement vs Traditional Logic')
    plt.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path_2 = os.path.join(plots_dir, f"savings_evolution_seed_{seed}.png")
    plt.savefig(save_path_2, dpi=300)
    print(f"Saved savings evolution to: {save_path_2}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize CEM Training Results")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the training results to plot")
    parser.add_argument("--res_dir", type=str, default="results", help="Directory containing history CSV files")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save images")
    
    args = parser.parse_args()
    plot_training_history(args.res_dir, args.seed, args.plots_dir)

if __name__ == "__main__":
    main()
