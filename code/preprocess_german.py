# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json
from datetime import timedelta

# ==========================================
# 1. 配置与路径
# ==========================================
BASE_DIR = "/Users/shiyinqi/Documents/code/energy/"
GEO_PATH = os.path.join(BASE_DIR, "00.data/raw/家庭经纬度信息.xlsx")
ENERGY_PATHS = [
    os.path.join(BASE_DIR, "00.data/raw/电量信息1-260415-260428.xlsx"),
    os.path.join(BASE_DIR, "00.data/raw/电量信息2.xlsx")
]
PRICE_PATH = os.path.join(BASE_DIR, "00.data/raw/电价260415~260428.xlsx")
WEATHER_PATH = os.path.join(BASE_DIR, "00.data/open-meteo-germany/data/open_meteo_germany.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "00.data/preprocessed/german_households_weather/")

# 窗口对齐配置
GOLDEN_START = "2026-04-15 02:00:00"
GOLDEN_END   = "2026-04-28 00:00:00"
TRAIN_LEN = 287 
PRICE_DIVISOR = 1.0 
SELL_PRICE_RATIO = 1.0 


# ==========================================
# 2. 辅助函数
# ==========================================
GERMAN_STATES = {
    'Bayern': [47.2, 50.6, 8.9, 13.8],
    'Baden-Württemberg': [47.5, 49.8, 7.5, 10.5],
    'Hessen': [49.3, 51.6, 8.2, 10.2],
    'Nordrhein-Westfalen': [50.3, 52.5, 5.8, 9.3],
    'Niedersachsen': [51.3, 54.0, 7.0, 11.5],
    'Rheinland-Pfalz': [48.9, 50.9, 6.2, 8.5],
    'Sachsen': [50.2, 51.7, 11.9, 15.0],
    'Sachsen-Anhalt': [51.3, 53.0, 10.5, 12.8],
    'Thüringen': [50.2, 51.6, 9.8, 12.6],
    'Brandenburg': [51.3, 53.6, 11.2, 14.8],
    'Berlin': [52.3, 52.7, 13.1, 13.8],
    'Schleswig-Holstein': [53.6, 55.1, 8.5, 11.5],
    'Hamburg': [53.4, 53.7, 9.7, 10.3],
    'Mecklenburg-Vorpommern': [53.1, 54.7, 10.5, 14.3],
    'Bremen': [53.0, 53.6, 8.4, 9.0],
    'Saarland': [49.1, 49.6, 6.3, 7.4]
}

def get_german_state(lat, lon):
    for state, bounds in GERMAN_STATES.items():
        lat_min, lat_max, lon_min, lon_max = bounds
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return state
    return 'Unknown'

def find_nearest_weather_grid(lat, lon, grid_points):
    distances = np.sqrt((grid_points[:, 0] - lat)**2 + (grid_points[:, 1] - lon)**2)
    return grid_points[np.argmin(distances)]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载天气数据
    print(">>> 正在加载并索引天气数据...")
    df_weather = pd.read_csv(WEATHER_PATH)
    # 将 UTC 时间转为德国本地时间 (UTC+2)
    df_weather['timestamp_local'] = pd.to_datetime(df_weather['time_utc']) + timedelta(hours=2)
    unique_grids = df_weather[['requested_latitude', 'requested_longitude']].drop_duplicates().values
    
    # 2. 加载家庭地理与电价数据
    print(">>> 正在加载原始电量与电价数据...")
    df_geo = pd.read_excel(GEO_PATH).dropna(subset=['id'])
    df_geo['id'] = df_geo['id'].astype(np.int64)
    german_geo = df_geo[df_geo['country'] == 'Germany'].copy()
    
    df_price = pd.read_excel(PRICE_PATH)
    df_price['timestamp'] = pd.to_datetime(df_price['startTime']).dt.floor('15min')
    df_price.set_index('timestamp', inplace=True)
    df_price = df_price.ffill().bfill()
    
    # 3. 加载电量数据并去重
    all_energy_dfs = []
    for path in ENERGY_PATHS:
        if os.path.exists(path):
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                all_energy_dfs.append(pd.read_excel(xl, sheet_name=sheet))
    df_energy_all = pd.concat(all_energy_dfs, ignore_index=True).dropna(subset=['plantId'])
    df_energy_all['plantId'] = df_energy_all['plantId'].astype(np.int64)
    df_energy_all['timestamp'] = pd.to_datetime(df_energy_all['statStartTime']).dt.floor('15min')
    df_energy_all = df_energy_all.drop_duplicates(subset=['plantId', 'timestamp'])
    
    print(f">>> 数据就绪。| 原始记录总行数: {len(df_energy_all)}")

    success_count = 0
    skip_log = []

    # 4. 家庭级处理循环
    for _, household in german_geo.iterrows():
        hid = household['id']
        lat, lon = household['latitude'], household['longitude']
        state = get_german_state(lat, lon)
        
        if state == 'Unknown':
            skip_log.append({'hid': hid, 'reason': 'Unknown State'})
            continue
            
        h_energy_raw = df_energy_all[df_energy_all['plantId'] == hid].copy()
        if h_energy_raw.empty: continue
            
        # 电量预处理与聚合
        h_energy_raw.set_index('timestamp', inplace=True)
        h_energy_raw = h_energy_raw[~h_energy_raw.index.duplicated(keep='first')]
        h_energy_15m = h_energy_raw[['pvSum', 'loadSum']].resample('15min').asfreq().interpolate(method='linear', limit=4)
        
        # 合并电价
        h_merged = h_energy_15m.join(df_price[['germany']], how='left')
        h_merged['germany'] = h_merged['germany'].ffill().bfill() / PRICE_DIVISOR
        
        # 聚合为小时
        h_hourly = h_merged.resample('h').agg({'pvSum': 'sum', 'loadSum': 'sum', 'germany': 'first'})
        h_hourly.rename(columns={'germany': 'price_eur_kwh'}, inplace=True)

        # 5. 注入天气数据
        nearest_grid = find_nearest_weather_grid(lat, lon, unique_grids)
        h_weather = df_weather[(df_weather['requested_latitude'] == nearest_grid[0]) & 
                               (df_weather['requested_longitude'] == nearest_grid[1])].copy()
        h_weather.set_index('timestamp_local', inplace=True)
        weather_features = ['temperature_2m', 'cloud_cover', 'shortwave_radiation', 'relative_humidity_2m', 'wind_speed_10m']
        h_hourly = h_hourly.join(h_weather[weather_features], how='left')

        # 6. 窗口裁剪
        h_hourly = h_hourly.sort_index().loc[GOLDEN_START:GOLDEN_END]
        
        # 插值补全细微空洞
        h_hourly = h_hourly.interpolate(method='linear', limit=2)

        # 生成电价预测窗 (shift 会在末尾产生 NaN)
        for i in range(1, 25):
            h_hourly[f'price_eur_kwh_t+{i}'] = h_hourly['price_eur_kwh'].shift(-i)
            
        # 生成天气预测窗 (未来 6 小时辐射)
        for i in range(1, 7):
            h_hourly[f'shortwave_radiation_t+{i}'] = h_hourly['shortwave_radiation'].shift(-i)

        h_hourly.dropna(inplace=True)
        if len(h_hourly) != TRAIN_LEN:
            skip_log.append({'hid': hid, 'reason': f'Length mismatch (Expected {TRAIN_LEN}, got {len(h_hourly)})'})
            continue
            
        if h_hourly['pvSum'].max() < 0.1: 
            skip_log.append({'hid': hid, 'reason': 'Minimal PV peak power'})
            continue

        # 时间与单位转换
        h_hourly['hour_of_day'] = h_hourly.index.hour
        h_hourly['day_of_week'] = h_hourly.index.dayofweek
        h_hourly['day_of_year'] = h_hourly.index.dayofyear
        h_hourly['day_of_month'] = h_hourly.index.day
        h_hourly['pv_gen_wh'] = h_hourly['pvSum'] * 1000.0
        h_hourly['user_load_wh'] = h_hourly['loadSum'] * 1000.0
        
        # 基准收益计算
        sell_p = h_hourly['price_eur_kwh'] * SELL_PRICE_RATIO
        pv_to_load_kwh = np.minimum(h_hourly['pv_gen_wh'], h_hourly['user_load_wh']) / 1000.0
        pv_to_grid_kwh = (h_hourly['pv_gen_wh'] - (pv_to_load_kwh * 1000.0)) / 1000.0
        grid_to_load_kwh = (h_hourly['user_load_wh'] - (pv_to_load_kwh * 1000.0)) / 1000.0
        h_hourly['baseline_reward'] = (pv_to_load_kwh * h_hourly['price_eur_kwh']) + \
                                      (pv_to_grid_kwh * sell_p) - \
                                      (grid_to_load_kwh * h_hourly['price_eur_kwh'])
        
        # 存盘
        h_hourly['lat'], h_hourly['lon'], h_hourly['state'] = lat, lon, state
        h_hourly['country'] = 'Germany'
        
        save_cols = ['pv_gen_wh', 'user_load_wh', 'price_eur_kwh'] + \
                    [f'price_eur_kwh_t+{i}' for i in range(1, 25)] + \
                    ['temperature_2m', 'cloud_cover', 'shortwave_radiation', 
                     'relative_humidity_2m', 'wind_speed_10m'] + \
                    [f'shortwave_radiation_t+{i}' for i in range(1, 7)] + \
                    ['hour_of_day', 'day_of_week', 'day_of_year', 'day_of_month', 'lat', 'lon', 'country', 'state', 'baseline_reward']
        
        h_hourly[save_cols].to_csv(os.path.join(OUTPUT_DIR, f"{hid}.csv"))
        success_count += 1
        if success_count % 20 == 0: print(f"进度: 已处理 {success_count} 户...")

    print(f"\n>>> 预处理完成。成功: {success_count} 户。数据存放于 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
