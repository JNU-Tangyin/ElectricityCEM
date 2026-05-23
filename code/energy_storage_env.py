import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import json

# ==========================================
# 电池物理与经济参数
# ==========================================
BATTERY_CAPACITY_WH = 4800 
MAX_CHARGE_POWER_W = 2400   
MAX_DISCHARGE_POWER_W = 2400
CHARGE_EFFICIENCY = 0.94
DISCHARGE_EFFICIENCY = 0.94
MIN_SOC = 0.1
MAX_SOC = 1.0
SELL_PRICE_RATIO = 1.0

class EnergyStorageEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data_input,
        episode_length_hours=168,
        sell_price_ratio=0.9,
        action_penalty_coef=0.0,
    ):
        super().__init__()
        self.episode_length_hours = episode_length_hours
        self.sell_price_ratio = sell_price_ratio
        self.action_penalty_coef = action_penalty_coef
        
        # 统一功率定义 (物理上限)
        self.max_charge_power_w = MAX_CHARGE_POWER_W
        self.max_discharge_power_w = MAX_DISCHARGE_POWER_W
        self.max_power_w = (MAX_CHARGE_POWER_W + MAX_DISCHARGE_POWER_W) / 2.0

        # 1. 加载数据
        if isinstance(data_input, str):
            self.df = pd.read_csv(data_input, index_col=0, parse_dates=True)
        else:
            self.df = data_input
            
        self.total_data_steps = len(self.df)
        self.price_col = 'price_eur_kwh'

        # 2. 计算局部归一化参数 (Local Min-Max)
        self.local_scaling = {}
        target_cols = ['pv_gen_wh', 'user_load_wh', 'price_eur_kwh', 
                       'shortwave_radiation']
        target_cols += [f'price_eur_kwh_t+{i}' for i in range(1, 25)]
        target_cols += [f'shortwave_radiation_t+{i}' for i in range(1, 7)]
        
        for col in target_cols:
            if col in self.df.columns:
                self.local_scaling[col] = {
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }

        # 3. 加载元数据
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))
        meta_path = os.path.join(base_dir, "00.data/preprocessed/german_metadata_weather.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(base_dir, "00.data/preprocessed/global_metadata.json")
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        self.all_states = self.meta.get('states', [])

        # --- Household Context ---
        self.lat = self.df['lat'].iloc[0]
        self.lon = self.df['lon'].iloc[0]
        self.local_state = self.df['state'].iloc[0] if 'state' in self.df.columns else 'Unknown'
        
        # 4. 定义空间
        # SOC(1) + PV/Load(2) + State(1) + Sin/Cos(2) + Price(26) + Weather(3) = 35
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(35,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 5. 状态变量
        self.current_soc = 0.5
        self.current_step = 0
        self.prev_pv = 0.0
        self.prev_load = 0.0

    def _get_obs(self):
        if self.current_step + 24 >= self.total_data_steps:
            return np.zeros(35, dtype=np.float32)

        row = self.df.iloc[self.current_step]
        obs = []

        def scale_val(val, col_name):
            s = self.local_scaling.get(col_name)
            if s and s['max'] > s['min']:
                return (val - s['min']) / (s['max'] - s['min'])
            return 0.0

        # [1] SOC (1)
        obs.append(self.current_soc)

        # [2] PV, Load (2)
        obs.append(scale_val(row['pv_gen_wh'], 'pv_gen_wh'))
        obs.append(scale_val(row['user_load_wh'], 'user_load_wh'))

        # [3] STATE Index (1)
        state_idx_val = 0.0
        if self.local_state in self.all_states:
            state_idx_val = self.all_states.index(self.local_state) / max(1, len(self.all_states) - 1)
        obs.append(state_idx_val)

        # [4] Time Features (2)
        h_rad = 2 * np.pi * row['hour_of_day'] / 24
        obs.append(np.sin(h_rad))
        obs.append(np.cos(h_rad))

        # [5] Price Features (26)
        # Global Level (1)
        g_price_min = self.meta['price_min']
        g_price_max = self.meta['price_max']
        g_price_range = g_price_max - g_price_min + 1e-9
        obs.append((row[self.price_col] - g_price_min) / g_price_range)
        # Local Window (25)
        obs.append(scale_val(row[self.price_col], 'price_eur_kwh'))
        for i in range(1, 25):
            obs.append(scale_val(row[f'price_eur_kwh_t+{i}'], f'price_eur_kwh_t+{i}'))

        # [6] Weather Features (3)
        obs.append(row['cloud_cover'] / 100.0) # 实时云量
        obs.append(scale_val(row['shortwave_radiation'], 'shortwave_radiation')) # 实时辐射
        obs.append(scale_val(row['shortwave_radiation_t+3'], 'shortwave_radiation_t+3')) # 3h预报

        return np.array(obs, dtype=np.float32)

    def get_normalization_factor(self):
        return self.df['user_load_wh'].mean() + 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and 'start_idx' in options:
            self.start_idx = options['start_idx']
        else:
            max_start = self.total_data_steps - self.episode_length_hours - 24
            self.start_idx = self.np_random.integers(0, max_start) if max_start > 0 else 0
        
        self.current_step = self.start_idx
        self.current_episode_end = min(self.start_idx + self.episode_length_hours, self.total_data_steps)
        self.current_soc = (MIN_SOC + MAX_SOC) / 2
        
        row = self.df.iloc[self.current_step]
        self.prev_pv = row['pv_gen_wh']
        self.prev_load = row['user_load_wh']
        
        return self._get_obs(), {}

    def step(self, action_norm):
        a = np.clip(action_norm[0], -1.0, 1.0)
        
        current_wh = self.current_soc * BATTERY_CAPACITY_WH
        max_wh_to_charge = (MAX_SOC * BATTERY_CAPACITY_WH - current_wh) / CHARGE_EFFICIENCY
        max_wh_to_discharge = (current_wh - MIN_SOC * BATTERY_CAPACITY_WH) * DISCHARGE_EFFICIENCY
        
        if a > 0:
            action_w = min(a * MAX_CHARGE_POWER_W, MAX_CHARGE_POWER_W, max_wh_to_charge)
        else:
            action_w = max(a * MAX_DISCHARGE_POWER_W, -MAX_DISCHARGE_POWER_W, -max_wh_to_discharge)
        
        row = self.df.iloc[self.current_step]
        buy_p = row[self.price_col]
        sell_p = buy_p * self.sell_price_ratio

        # 执行物理动作
        delta_wh = 0
        if action_w > 0: 
            delta_wh = action_w * CHARGE_EFFICIENCY
        elif action_w < 0: 
            delta_wh = action_w / DISCHARGE_EFFICIENCY

        old_wh = current_wh
        new_wh = np.clip(old_wh + delta_wh, MIN_SOC * BATTERY_CAPACITY_WH, MAX_SOC * BATTERY_CAPACITY_WH)
        actual_delta_wh = new_wh - old_wh
        
        e_from_batt = abs(actual_delta_wh) * DISCHARGE_EFFICIENCY if actual_delta_wh < 0 else 0.0
        e_into_batt = actual_delta_wh / CHARGE_EFFICIENCY if actual_delta_wh > 0 else 0.0
        self.current_soc = new_wh / BATTERY_CAPACITY_WH

        # 能源平衡逻辑
        pv, load = row['pv_gen_wh'], row['user_load_wh']
        pv_to_load = min(pv, load)
        rem_pv, rem_load = pv - pv_to_load, load - pv_to_load
        pv_to_batt = min(rem_pv, e_into_batt)
        rem_pv -= pv_to_batt
        grid_buy_for_batt = e_into_batt - pv_to_batt
        batt_to_load = min(rem_load, e_from_batt)
        rem_load -= batt_to_load
        batt_to_grid = e_from_batt - batt_to_load 
        
        # Reward = x4 + x3 + x2 - x1
        x1_cost = ((load - pv_to_load - batt_to_load) + grid_buy_for_batt) / 1000.0 * buy_p
        x2_revenue = (rem_pv + batt_to_grid) / 1000.0 * sell_p
        x3_savings_pv = pv_to_load / 1000.0 * buy_p
        x4_savings_batt = batt_to_load / 1000.0 * buy_p
        
        action_penalty = -(
            abs(action_w) / self.max_power_w
        ) * self.action_penalty_coef
        
        reward = x4_savings_batt + x3_savings_pv + x2_revenue - x1_cost + action_penalty

        self.prev_pv = pv
        self.prev_load = load
        self.current_step += 1
        
        done = self.current_step >= self.current_episode_end
        return self._get_obs(), reward, done, False, {
            "profit": reward,
            "soc": self.current_soc,
            "price": buy_p
        }
    
    def close(self):
        pass
