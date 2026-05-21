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
SELL_PRICE_RATIO = 0.9 

class EnergyStorageEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data_input, episode_length_hours=168, sell_price_ratio=SELL_PRICE_RATIO):
        super().__init__()
        self.episode_length_hours = episode_length_hours
        self.sell_price_ratio = sell_price_ratio

        # 1. 加载数据
        if isinstance(data_input, str):
            self.df = pd.read_csv(data_input, index_col=0, parse_dates=True)
        else:
            self.df = data_input
            
        self.total_data_steps = len(self.df)
        self.price_col = 'price_eur_kwh'

        # 2.计算局部归一化参数 (Local Min-Max)
        self.local_scaling = {}
        target_cols = ['pv_gen_wh', 'user_load_wh', 'price_eur_kwh', 
                       'temperature_2m', 'cloud_cover', 'shortwave_radiation', 
                       'relative_humidity_2m', 'wind_speed_10m']
        # 包含所有预测窗列
        target_cols += [f'price_eur_kwh_t+{i}' for i in range(1, 25)]
        target_cols += [f'shortwave_radiation_t+{i}' for i in range(1, 7)]
        
        for col in target_cols:
            if col in self.df.columns:
                self.local_scaling[col] = {
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }

        # 3. 加载德国元数据 (仅用于地理位置归一化参考)
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
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(52,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 5. 状态变量
        self.current_soc = 0.5
        self.current_step = 0
        self.prev_pv = 0.0
        self.prev_load = 0.0

    def _get_obs(self):
        if self.current_step + 24 >= self.total_data_steps:
            return np.zeros(52, dtype=np.float32)

        row = self.df.iloc[self.current_step]
        obs = []

        def scale_val(val, col_name):
            s = self.local_scaling.get(col_name)
            if s and s['max'] > s['min']:
                return (val - s['min']) / (s['max'] - s['min'])
            return 0.0

        # [1] SOC (1)
        obs.append(self.current_soc)

        # [2] Geographical Context (2)
        obs.append((self.lat - 47.0) / 8.0) 
        obs.append((self.lon - 5.0) / 10.0) 

        # [3] Household Scale Signatures (2)
        obs.append(self.df['pv_gen_wh'].max() / self.meta['pv_max'])
        obs.append(self.df['user_load_wh'].max() / self.meta['load_max'])

        # [4] STATE Index (1)
        state_idx_val = 0.0
        if self.local_state in self.all_states:
            state_idx_val = self.all_states.index(self.local_state) / max(1, len(self.all_states) - 1)
        obs.append(state_idx_val)

        # [5] Real-time Energy (4)
        obs.append(scale_val(row['pv_gen_wh'], 'pv_gen_wh'))
        obs.append(scale_val(row['user_load_wh'], 'user_load_wh'))
        obs.append(np.clip((row['pv_gen_wh'] - self.prev_pv)/500.0, -1, 1))
        obs.append(np.clip((row['user_load_wh'] - self.prev_load)/500.0, -1, 1))

        # [6] Time Features (5)
        h_rad = 2 * np.pi * row['hour_of_day'] / 24
        obs.append(np.sin(h_rad))
        obs.append(np.cos(h_rad))
        obs.append(row['day_of_week'] / 7.0)
        obs.append(row['day_of_year'] / 365.0)
        obs.append(row['day_of_month'] / 31.0)

        # [7] Price Features (26 dimensions)
        # [7.1] Global Price Level (1 dim) 
        g_price_min = self.meta['price_min']
        g_price_max = self.meta['price_max']
        g_price_range = g_price_max - g_price_min + 1e-9
        obs.append((row[self.price_col] - g_price_min) / g_price_range)

        # [7.2] Local Window (25 dims) - 局部 Min-Max
        obs.append(scale_val(row[self.price_col], 'price_eur_kwh'))
        for i in range(1, 25):
            obs.append(scale_val(row[f'price_eur_kwh_t+{i}'], f'price_eur_kwh_t+{i}'))

        # [8] Weather Features (5 dimensions)
        obs.append(scale_val(row['temperature_2m'], 'temperature_2m'))
        obs.append(row['cloud_cover'] / 100.0)
        obs.append(row['relative_humidity_2m'] / 100.0)
        obs.append(scale_val(row['wind_speed_10m'], 'wind_speed_10m'))
        obs.append(scale_val(row['shortwave_radiation'], 'shortwave_radiation'))

        # [9] Radiation Forecast (6 dimensions)
        for i in range(1, 7):
            obs.append(scale_val(row[f'shortwave_radiation_t+{i}'], f'shortwave_radiation_t+{i}'))

        return np.array(obs, dtype=np.float32)

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
        # 物理层限制：根据当前 SOC 动态约束动作空间
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

        # 能源平衡逻辑 (PV/Load/Grid)
        pv, load = row['pv_gen_wh'], row['user_load_wh']
        pv_to_load = min(pv, load)
        rem_pv, rem_load = pv - pv_to_load, load - pv_to_load
        pv_to_batt = min(rem_pv, e_into_batt)
        rem_pv -= pv_to_batt
        grid_buy_for_batt = e_into_batt - pv_to_batt
        batt_to_load = min(rem_load, e_from_batt)
        rem_load -= batt_to_load
        batt_to_grid = e_from_batt - batt_to_load 
        
        # x1 = 买电总支出 (家电缺口 + 充电池买电)
        x1_cost = ((load - pv_to_load - batt_to_load) + grid_buy_for_batt) / 1000.0 * buy_p
        # x2 = 卖电总收入 (光伏余电卖给电网 + 电池余电卖给电网)
        x2_revenue = (rem_pv + batt_to_grid) / 1000.0 * sell_p
        # x3 = 光伏直接供给家电省下的钱 (避免买电)
        x3_savings_pv = pv_to_load / 1000.0 * buy_p
        # x4 = 电池供给家电省下的钱 (避免买电)
        x4_savings_batt = batt_to_load / 1000.0 * buy_p
        
        reward = x4_savings_batt + x3_savings_pv + x2_revenue - x1_cost

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
