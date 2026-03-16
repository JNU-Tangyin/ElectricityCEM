import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

# Real Battery Parameters (Provided by client)
BATTERY_CAPACITY_WH = 4800  # Wh (4.8 kWh)
MAX_CHARGE_POWER_W = 2400    # W (2.4 kW)
MAX_DISCHARGE_POWER_W = 2400 # W (2.4 kW)
CHARGE_EFFICIENCY = 0.94
DISCHARGE_EFFICIENCY = 0.94
MIN_SOC = 0.1 # 10%
MAX_SOC = 1.0 # 100%

class EnergyStorageEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, data_path='processed_energy_data_germany.csv', episode_length_hours=24, render_mode=None, sell_price_ratio=0.9):
        super().__init__()
        self.data_path = data_path
        self.render_mode = render_mode
        self.episode_length_hours = episode_length_hours
        self.sell_price_ratio = sell_price_ratio

        # Load preprocessed data
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        self.total_data_steps = len(self.df) # FIX: Total steps in the dataset

        # FIX: Calculate scaling parameters for observation space
        self.scaling_params = {}
        # PV, Load, Price (current and future)
        # Price (current and future) columns from self.df
        price_cols = ['price_de_eur'] + [f'price_de_eur_t+{i}' for i in range(1, 24)]
        # We need min/max for all these price columns, but they are all based on 'price_de_eur'
        # So we use the min/max of the base 'price_de_eur' for all price scaling
        self.scaling_params['price_de_eur'] = {'min': self.df['price_de_eur'].min(), 'max': self.df['price_de_eur'].max()}

        for col in ['pv_gen_wh', 'user_load_wh']:
            self.scaling_params[col] = {'min': self.df[col].min(), 'max': self.df[col].max()}
        
        # Time features
        self.scaling_params['hour_of_day'] = {'min': 0, 'max': 23}
        self.scaling_params['day_of_week'] = {'min': 0, 'max': 6}
        self.scaling_params['day_of_year'] = {'min': self.df['day_of_year'].min(), 'max': self.df['day_of_year'].max()} # Based on actual data range
        self.scaling_params['month'] = {'min': 1, 'max': 12}
        self.scaling_params['day_of_month'] = {'min': 1, 'max': 31}
        # SOC is already 0-1, no need to scale

        # Define Observation Space
        # FIX: Corrected calculation for num_features after encoding hour_of_day as sin/cos
        # 1 (SOC) + 1 (PV) + 1 (Load) + 2 (Sin/Cos Hour) + 4 (Other Time Features) + 24 (Prices) = 33
        num_features = 1 + 1 + 1 + 2 + 4 + 24 # SOC + PV + Load + Sin/Cos Hour + DayOfWeek/Year/Month/DayOfMonth + Prices
        # After normalization, all features will be in [0, 1] range
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_features,), dtype=np.float32)

        # Define Action Space: Continuous power for charge/discharge
        # Action could be a value between -MAX_DISCHARGE_POWER_W and +MAX_CHARGE_POWER_W
        self.action_space = spaces.Box(low=-MAX_DISCHARGE_POWER_W, high=MAX_CHARGE_POWER_W, shape=(1,), dtype=np.float32)

        # Initial SOC
        self.current_soc = (MIN_SOC + MAX_SOC) / 2 # Start in the middle of operational range

    def _get_obs(self):
        # FIX: _get_obs should use self.current_step (which is the actual index in self.df)
        # Ensure that current_step is within bounds, especially when accessing future prices
        if self.current_step + 23 >= self.total_data_steps: # Need to check up to t+23 price
            # This case should ideally be handled by termination, but as a safeguard.
            # If we reach here, it means the episode was not terminated correctly.
            return np.zeros(self.observation_space.shape[0])

        current_row = self.df.iloc[self.current_step]
        
        # Extract features for the observation
        # Ensure the order and type match observation_space definition
        obs = []

        # SOC (already normalized)
        obs.append(self.current_soc)

        # PV, Load features (apply min-max scaling)
        for col in ['pv_gen_wh', 'user_load_wh']:
            min_val = self.scaling_params[col]['min']
            max_val = self.scaling_params[col]['max']
            # Handle max_val == min_val to prevent division by zero for constant features
            if max_val == min_val:
                scaled_val = 0.0 # If all values are the same, scaled value is 0
            else:
                scaled_val = (current_row[col] - min_val) / (max_val - min_val)
            obs.append(scaled_val)
        
        # Time features (apply min-max scaling)
        # FIX: Encode hour_of_day using sin/cos for periodicity
        hour_rad = 2 * np.pi * current_row['hour_of_day'] / 24
        obs.append(np.sin(hour_rad))
        obs.append(np.cos(hour_rad))
        obs.append((current_row['day_of_week'] - self.scaling_params['day_of_week']['min']) / (self.scaling_params['day_of_week']['max'] - self.scaling_params['day_of_week']['min'] + 1e-7))
        obs.append((current_row['day_of_year'] - self.scaling_params['day_of_year']['min']) / (self.scaling_params['day_of_year']['max'] - self.scaling_params['day_of_year']['min'] + 1e-7))
        obs.append((current_row['month'] - self.scaling_params['month']['min']) / (self.scaling_params['month']['max'] - self.scaling_params['month']['min'] + 1e-7))
        obs.append((current_row['day_of_month'] - self.scaling_params['day_of_month']['min']) / (self.scaling_params['day_of_month']['max'] - self.scaling_params['day_of_month']['min'] + 1e-7))

        # Add current price and future prices (total 24 price points) - apply min-max scaling using global price_de_eur min/max
        min_price_val = self.scaling_params['price_de_eur']['min']
        max_price_val = self.scaling_params['price_de_eur']['max']
        
        for i in range(24): 
            col_name = 'price_de_eur' if i == 0 else f'price_de_eur_t+{i}'
            if max_price_val == min_price_val:
                scaled_val = 0.0
            else:
                scaled_val = (current_row[col_name] - min_price_val) / (max_price_val - min_price_val)
            obs.append(scaled_val)

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        # Optional: Return auxiliary information
        # FIX: _get_info should use self.current_step
        if self.current_step >= self.total_data_steps:
             return { "error": "Accessed info beyond data limit" } # Safeguard

        current_row = self.df.iloc[self.current_step]
        # Ensure strategy_mode is present before trying to access it
        strategy_mode = current_row['strategy_mode'] if 'strategy_mode' in current_row.index else -1 # Default to -1 if not present
        return {
            "current_time": current_row.name, # Use row.name for timestamp
            "price": current_row['price_de_eur'],
            "pv_gen": current_row['pv_gen_wh'],
            "user_load": current_row['user_load_wh'],
            "soc": self.current_soc,
            # Add 24-hour unscaled price forecast for company baseline
            "price_forecast_24h": [current_row['price_de_eur']] + [current_row[f'price_de_eur_t+{i}'] for i in range(1, 24)],
            "strategy_mode": strategy_mode # Add the strategy mode
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # FIX: Re-enable random starting point for episodes to improve generalization (after confirming deterministic performance)
        # Pick a random starting point in the dataset for the episode
        # Ensure there's enough data for a full episode_length_hours
        # We need self.df.iloc[self.start_idx + self.episode_length_hours - 1] to be valid.
        # Also, the observation at start_idx must have valid future prices up to t+23.
        # So, the end of the episode (self.start_idx + self.episode_length_hours) should not go beyond (total_data_steps - 24)
        
        max_valid_start_for_episode = self.total_data_steps - self.episode_length_hours - 23 # Last possible index for start_idx
        
        if max_valid_start_for_episode <= 0: # Not enough data for one episode
            self.start_idx = 0 # Start from beginning
            print(f"Warning: Not enough data for episode_length_hours={self.episode_length_hours}, starting from index 0.")
        else:
            # Randomly select a starting index for the episode
            self.start_idx = self.np_random.integers(0, max_valid_start_for_episode)
        
        self.current_step = self.start_idx # Current step for the environment starts at this random index
        self.current_episode_end_step = self.start_idx + self.episode_length_hours # Define episode end for termination
        self.current_soc = (MIN_SOC + MAX_SOC) / 2 # Resets battery SOC to initial state

        observation = self._get_obs() # Gets the initial observation
        info = self._get_info() # Gets initial info
        return observation, info # Returns the initial state and info

    def step(self, action):
        # Ensure action is within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)[0] # Action is a single value

        current_row = self.df.iloc[self.current_step]
        
        price_de_eur = current_row['price_de_eur']
        pv_gen_wh = current_row['pv_gen_wh']
        user_load_wh = current_row['user_load_wh']

        # 定义买电价格和卖电价格
        buy_price = price_de_eur
        sell_price = price_de_eur * self.sell_price_ratio

        # --- Simulate Battery and Energy Flow ---
        # Action is power in W. We are simulating for 1 hour, so Power * 1h = Energy in Wh
        action_wh = action # Since action is in W and time step is 1 hour

        # Calculate potential next SOC based on action (ignoring min/max for now)
        delta_soc_wh = 0
        if action_wh > 0: # Charging
            charge_amount_wh = min(action_wh, MAX_CHARGE_POWER_W)
            delta_soc_wh = charge_amount_wh * CHARGE_EFFICIENCY
        elif action_wh < 0: # Discharging
            discharge_amount_wh = min(abs(action_wh), MAX_DISCHARGE_POWER_W)
            delta_soc_wh = -discharge_amount_wh / DISCHARGE_EFFICIENCY

        # Apply SOC limits (hard constraints)
        potential_next_soc_wh = self.current_soc * BATTERY_CAPACITY_WH + delta_soc_wh
        
        # Clip SOC to operational range
        next_soc_wh = np.clip(potential_next_soc_wh, MIN_SOC * BATTERY_CAPACITY_WH, MAX_SOC * BATTERY_CAPACITY_WH)
        
        # Calculate actual energy charged/discharged after clipping
        actual_delta_soc_wh = next_soc_wh - (self.current_soc * BATTERY_CAPACITY_WH) # This is the actual change in stored energy (Wh)

        energy_from_battery_wh = 0.0 # Energy effectively supplied by battery to load/grid (after discharge efficiency)
        energy_into_battery_wh = 0.0 # Energy effectively consumed by battery from PV/grid (before charge efficiency)

        if actual_delta_soc_wh < 0: # Battery actually discharged (SOC decreased)
            # abs(actual_delta_soc_wh) is the amount of energy *removed from* battery storage
            energy_from_battery_wh = abs(actual_delta_soc_wh) * DISCHARGE_EFFICIENCY # Useful energy provided
            
        elif actual_delta_soc_wh > 0: # Battery actually charged (SOC increased)
            # actual_delta_soc_wh is the amount of energy *added to* battery storage
            energy_into_battery_wh = actual_delta_soc_wh / CHARGE_EFFICIENCY # Energy drawn from source
            
        # Update current SOC
        self.current_soc = next_soc_wh / BATTERY_CAPACITY_WH

        # --- Calculate Grid Interaction & Reward ---
        # Reward calculation: x4 + x3 + x2 - x1
        # x1 = cost of buying from grid
        # x2 = revenue from selling to grid
        # x3 = avoided cost from PV used for load
        # x4 = avoided cost from battery used for load

        # --- Energy Flow Accounting (Revised for Realistic Arbitrage) ---
        # 1. Satisfy user load with PV generation first
        load_satisfied_by_pv_wh = min(user_load_wh, pv_gen_wh)
        remaining_load_to_satisfy_wh = user_load_wh - load_satisfied_by_pv_wh
        remaining_pv_generation_wh = pv_gen_wh - load_satisfied_by_pv_wh

        # 2. Use surplus PV to charge battery if charging is requested (energy_into_battery_wh)
        # This is "Free" charging.
        pv_to_battery_wh = min(remaining_pv_generation_wh, energy_into_battery_wh)
        remaining_pv_generation_wh -= pv_to_battery_wh
        # Remaining charging demand must be met by grid purchase
        grid_buy_for_battery_charge_wh = energy_into_battery_wh - pv_to_battery_wh

        # 3. Satisfy remaining load with battery discharge
        load_satisfied_by_battery_wh = min(remaining_load_to_satisfy_wh, energy_from_battery_wh)
        remaining_load_to_satisfy_wh -= load_satisfied_by_battery_wh
        # If action was to discharge but load is already satisfied, the rest is sold to grid
        remaining_discharge_available_wh = energy_from_battery_wh - load_satisfied_by_battery_wh

        # 4. Final grid interactions
        load_satisfied_by_grid_wh = remaining_load_to_satisfy_wh # Remaining load gap
        grid_sell_from_pv_wh = remaining_pv_generation_wh # Surplus PV after load and charging
        grid_sell_from_battery_wh = remaining_discharge_available_wh # Surplus battery discharge

        # --- Calculate components for Reward (Benefit-Cost Framework) ---
        # x1: Actual cost of buying from grid
        x1 = (load_satisfied_by_grid_wh + grid_buy_for_battery_charge_wh) * buy_price

        # x2: Actual revenue from selling to grid
        x2 = (grid_sell_from_pv_wh + grid_sell_from_battery_wh) * sell_price

        # x3: Avoided cost by using PV directly for load (avoid buying at buy_price)
        x3 = load_satisfied_by_pv_wh * buy_price

        # x4: Avoided cost by using Battery directly for load (avoid buying at buy_price)
        x4 = load_satisfied_by_battery_wh * buy_price

        # Total Reward: (Savings from PV/Battery) + (Revenue) - (Cost)
        reward = x4 + x3 + x2 - x1

        # --- Transition to next state ---
        self.current_step += 1
        # Check if episode terminated (reached end of current episode duration)
        terminated = self.current_step >= self.current_episode_end_step 
        truncated = False 

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            # Implement rendering if needed, e.g., plot SOC, prices, actions
            pass

    def close(self):
        # Clean up resources
        pass

if __name__ == '__main__':
    # Example usage:
    env = EnergyStorageEnv(data_path='processed_energy_data_germany.csv')
    
    obs, info = env.reset()
    print(f"Initial Observation: {obs}")
    print(f"Initial Info: {info}")

    # Take a random action (e.g., charge 1kW)
    action = np.array([1000.0]) # Example: charge 1000W (1000Wh over 1 hour)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step 1:")
    print(f"Action: {action}")
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Info: {info}")

    # Take another random action (e.g., discharge 500W)
    action = np.array([-500.0]) # Example: discharge 500W (500Wh over 1 hour)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step 2:")
    print(f"Action: {action}")
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Info: {info}")

    # Loop for a few steps
    total_reward = 0
    for _ in range(5):
        action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {env.current_step}, Action: {action[0]:.2f}, Reward: {reward:.2f}, SOC: {info['soc']:.2f}, Price: {info['price']:.2f}, PV: {info['pv_gen']:.2f}, Load: {info['user_load']:.2f}")
        if terminated or truncated:
            break
    print(f"Total reward over 5 steps: {total_reward:.2f}")

    env.close()