import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import json

# ==========================================
# 电池物理与经济参数 (全局常量)
# 这些参数由公司提供，是固定硬件规格。
# ==========================================
BATTERY_CAPACITY_WH = 4800 
MAX_CHARGE_POWER_W = 2400   
MAX_DISCHARGE_POWER_W = 2400
CHARGE_EFFICIENCY = 0.94
DISCHARGE_EFFICIENCY = 0.94
MIN_SOC = 0.1
MAX_SOC = 1.0

# ==========================================
# 物理仿真 (Shared Physics)
# ==========================================
def apply_battery_physics(soc, intended_power_w):
    """
    仿真电池状态转换，严格执行功率和容量限制。
    设计逻辑：该函数将 AI 的“意图动作”转化为“实际可执行动作”。
    考虑到充放电损耗，我们在计算 SOC 变化时分别应用了充电和放电效率，
    以反映真实电化学过程中的能量耗散。
    """
    current_wh = soc * BATTERY_CAPACITY_WH
    
    if intended_power_w > 0: 
        # 充电时：考虑充电效率，计算电池内部实际接收到的能量。
        # 同时限制充电功率不得超过硬件上限或电池剩余容量。
        max_wh_to_charge = (MAX_SOC * BATTERY_CAPACITY_WH - current_wh) / CHARGE_EFFICIENCY
        actual_port_w = min(intended_power_w, MAX_CHARGE_POWER_W, max_wh_to_charge)
        delta_wh_internal = actual_port_w * CHARGE_EFFICIENCY
    elif intended_power_w < 0: 
        # 放电时：考虑放电效率，电池内部实际消耗的能量通常大于输出到负载的能量。
        max_wh_to_discharge = (current_wh - MIN_SOC * BATTERY_CAPACITY_WH) * DISCHARGE_EFFICIENCY
        actual_port_w = max(intended_power_w, -MAX_DISCHARGE_POWER_W, -max_wh_to_discharge)
        delta_wh_internal = actual_port_w / DISCHARGE_EFFICIENCY
    else:
        actual_port_w = 0.0
        delta_wh_internal = 0.0

    # 最终通过 np.clip 确保 SOC 永远不会由于数值精度问题超出 10%-100% 的安全运行区间。
    new_soc = np.clip((current_wh + delta_wh_internal) / BATTERY_CAPACITY_WH, MIN_SOC, MAX_SOC)
    return new_soc, actual_port_w

# ==========================================
# 统一内核函数：财务结算 (Shared Economics)
# 实现公司定义的 Benefit-Cost 结算模型。
# ==========================================
def calculate_economics(pv_wh, load_wh, actual_port_w, buy_price, sell_ratio):
    """
    计算当前时间步的净经济收益。
    设计逻辑：通过追踪 PV 直接满足负载、电池满足负载、以及并网交易的电量流向，
    量化不同策略下的收益差异。这引导 AI 学习在电价高时放电、在电价低或有余电时充电。
    """
    sell_price = buy_price * sell_ratio
    e_in = max(0.0, actual_port_w)
    e_out = max(0.0, -actual_port_w)

    # 能量分配优先级逻辑：
    # 1. PV 优先满足家庭负载。
    pv_to_load = min(pv_wh, load_wh)
    rem_pv, rem_load = pv_wh - pv_to_load, load_wh - pv_to_load
    
    # 2. 余电优先给电池充电。
    pv_to_batt = min(rem_pv, e_in)
    grid_buy_batt = e_in - pv_to_batt
    rem_pv -= pv_to_batt
    
    # 3. 负载缺口由电池放电满足。
    batt_to_load = min(rem_load, e_out)
    batt_to_grid = e_out - batt_to_load 
    rem_load -= batt_to_load
    
    # 经济收益的四个组成部分：
    # s_pv: PV 自发自用节省的购电费。
    # s_bt: 电池放电自用节省的购电费。
    # r_ex: 向电网卖电产生的收入。
    # c_im: 从电网买电产生的实际成本。
    s_pv = pv_to_load / 1000.0 * buy_price
    s_bt = batt_to_load / 1000.0 * buy_price
    r_ex = (rem_pv + batt_to_grid) / 1000.0 * sell_price
    c_im = (rem_load + grid_buy_batt) / 1000.0 * buy_price
    
    return s_pv + s_bt + r_ex - c_im

class EnergyStorageEnv(gym.Env):
    """
    为德国家庭定制的强化学习环境。
    通过 35 维的状态观测空间，为线性策略(Linear Policy)提供了足够丰富的
    上下文信息，使其能够像非线性优化器一样捕捉复杂的价格波动和天气模式。
    """
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data_input,
        episode_length_hours=168,
        sell_price_ratio=0.9,
        action_penalty_coef=0.0,
        meta_path=None
    ):
        super().__init__()
        self.episode_length_hours = episode_length_hours
        self.sell_price_ratio = sell_price_ratio
        self.action_penalty_coef = action_penalty_coef
        self.max_power_w = (MAX_CHARGE_POWER_W + MAX_DISCHARGE_POWER_W) / 2.0

        if isinstance(data_input, pd.DataFrame):
            self.df = data_input
        else:
            self.df = pd.read_csv(data_input, index_col=0, parse_dates=True)
            
        self.total_data_steps = len(self.df)
        self.price_col = 'price_eur_kwh'

        # 状态特征配置：包含了影响决策的所有关键变量。
        # 注意：包含了未来 24 小时的预测电价，因为这在欧洲市场是已知信息，
        # 也是 AI 能够进行前瞻性调度(如提前充电以应对电价高峰)的核心前提。
        required_cols = ['pv_gen_wh', 'user_load_wh', 'price_eur_kwh', 'shortwave_radiation', 'shortwave_radiation_t+3', 'cloud_cover']
        required_cols += [f'price_eur_kwh_t+{i}' for i in range(1, 25)]
        
        # 局部归一化：为了让 AI 对特定家庭的功率波动保持高度的数值敏感度。
        # 如果统一使用极大的全局范围，某些小功率家庭的特征可能会被压缩到极小区间。
        self.local_scaling = {}
        for col in required_cols:
            if col != 'cloud_cover':
                self.local_scaling[col] = {'min': self.df[col].min(), 'max': self.df[col].max()}

        # 元数据加载(包含路径修复与容错)：用于获取跨家庭的全局特征(如地理位置、全局电价边界)。
        # 这确保了模型在处理不同家庭时，对于“电价高低”有一个统一的基准认知。
        if meta_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(script_dir))
            possible_names = ["german_metadata.json", "global_metadata.json", "german_metadata_weather.json"]
            for name in possible_names:
                p = os.path.join(base_dir, "00.data/preprocessed", name)
                if os.path.exists(p):
                    meta_path = p
                    break
        
        if meta_path is None or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Could not find any metadata file in 00.data/preprocessed/. Tried: {possible_names}")
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        self.all_states = self.meta.get('states', [])

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(35,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.current_soc = 0.5
        self.current_step = 0

    def _get_obs(self):
        """
        构建状态向量。
        设计逻辑：将局部敏感特征 (PV/负载)与全局参考特征(电价)相结合，
        并利用 Sin/Cos 编码处理小时特征，以保留 24 小时循环的周期性语义。
        """
        row = self.df.iloc[self.current_step]
        obs = []
        def scale(val, name):
            s = self.local_scaling.get(name)
            return (val - s['min']) / (s['max'] - s['min']) if s and s['max'] > s['min'] else 0.0

        obs.append(self.current_soc)
        obs.append(scale(row['pv_gen_wh'], 'pv_gen_wh'))
        obs.append(scale(row['user_load_wh'], 'user_load_wh'))
        
        # 地理位置特征：将联邦州索引映射到数值，帮助模型区分不同地区的辐射和价格习惯。
        s_idx = 0.0
        local_s = self.df['state'].iloc[0]
        if local_s in self.all_states:
            s_idx = self.all_states.index(local_s) / max(1, len(self.all_states) - 1)
        obs.append(s_idx)

        # 周期性特征编码：防止 23 点和 0 点在模型输入中出现突变。
        h_rad = 2 * np.pi * row['hour_of_day'] / 24
        obs.append(np.sin(h_rad))
        obs.append(np.cos(h_rad))

        # 全局价格参考：让模型知道当前电价在全德国范围内的相对水平。
        g_min, g_max = self.meta['price_min'], self.meta['price_max']
        obs.append((row[self.price_col] - g_min) / (g_max - g_min + 1e-9))
        
        # 局部价格序列：捕捉当前家庭正在经历的价格趋势和波动窗口。
        obs.append(scale(row[self.price_col], 'price_eur_kwh'))
        for i in range(1, 25): obs.append(scale(row[f'price_eur_kwh_t+{i}'], f'price_eur_kwh_t+{i}'))

        obs.append(row['cloud_cover'] / 100.0)
        obs.append(scale(row['shortwave_radiation'], 'shortwave_radiation'))
        obs.append(scale(row['shortwave_radiation_t+3'], 'shortwave_radiation_t+3'))
        
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = options.get('start_idx', 0) if options else 0
        self.current_episode_end = min(self.current_step + self.episode_length_hours, self.total_data_steps)
        self.current_soc = (MIN_SOC + MAX_SOC) / 2
        return self._get_obs(), {}

    def step(self, action_norm):
        """
        执行一个时间步(1小时)。
        将 AI 策略输出的归一化动作(-1 到 1)映射到物理功率，
        并调用统一的物理和经济内核，确保仿真精度和奖励信号的真实性。
        """
        a = np.clip(action_norm[0], -1.0, 1.0)
        row = self.df.iloc[self.current_step]
        
        # 映射动作并委托给统一内核。
        intended_w = a * MAX_CHARGE_POWER_W if a > 0 else a * MAX_DISCHARGE_POWER_W
        self.current_soc, actual_port_w = apply_battery_physics(self.current_soc, intended_w)
        
        actual_profit = calculate_economics(
            row['pv_gen_wh'], row['user_load_wh'], 
            actual_port_w, row['price_eur_kwh'], self.sell_price_ratio
        )
        
        # 动作惩罚项：默认为0。可选配置，用于抑制过于频繁或极端的动作，保护电池寿命。
        action_penalty = -(abs(actual_port_w) / self.max_power_w) * self.action_penalty_coef
        reward = actual_profit + action_penalty

        self.current_step += 1
        done = self.current_step >= self.current_episode_end
        return self._get_obs(), reward, done, False, {
            "profit": actual_profit, "soc": self.current_soc
        }
    
    def close(self): pass
