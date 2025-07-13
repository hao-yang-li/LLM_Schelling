import sys
import os
import random
import numpy as np
import pandas as pd
import json
import yaml

# Ensure the root directory is in the system path to find the 'src' and 'constant' modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.map import Map
from constant import MAP_SCOPE, MAP_DATA_PATH

# --- Configuration ---
CITY = 'Chicago'
TOTAL_AGENTS_TO_GENERATE = 1000  # 您可以调整希望生成的总Agent数量
POVERTY_DATA_PATH = 'dataset/01_Poverty_CBG_level.csv' # 收入数据路径

def load_config():
    """加载YAML配置文件。"""
    with open("config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_income_distribution_from_median(median_income: float) -> dict:
    """
    根据家庭收入中位数生成一个倾斜的收入水平分布。
    这个函数会确定中位数所在的收入等级，并提高该等级的概率。
    """
    # 定义收入等级的上限
    BRACKETS = {
        'Low': 30000,
        'Lower_Middle': 60000,
        'Middle': 100000,
        'Upper_Middle': 150000,
        'High': float('inf')
    }
    # 一个基础的、平均的概率分布
    base_probs = {'Low': 0.15, 'Lower_Middle': 0.25, 'Middle': 0.30, 'Upper_Middle': 0.20, 'High': 0.10}

    # 找出中位数所在的等级
    agent_majority_bracket = 'Low'
    for bracket, upper_bound in BRACKETS.items():
        if median_income < upper_bound:
            agent_majority_bracket = bracket
            break
            
    # 调整分布：给中位数所在的等级增加权重
    skewed_probs = base_probs.copy()
    skewed_probs[agent_majority_bracket] += 0.25  # 显著提高权重

    # 归一化，确保所有概率加起来等于1
    total = sum(skewed_probs.values())
    normalized_probs = {k: v / total for k, v in skewed_probs.items()}
    
    return normalized_probs

def main():
    """
    Main function to generate agents in a random subset of CBGs.
    """
    # --- 加载配置 ---
    config = load_config()
    schelling_params = config.get('schelling_model', {})
    num_cbgs_to_select = schelling_params.get('num_blocks_to_populate', 10) # 默认为10
    altruist_ratio = schelling_params.get('altruist_ratio', 0.1) # 默认为0.1
    
    print(f"--- Starting agent generation for a subset of {CITY} using config ---")
    print(f"Number of CBGs to populate: {num_cbgs_to_select}")
    print(f"Altruist Ratio: {altruist_ratio}")

    # --- 0. Load Income Data ---
    try:
        income_df = pd.read_csv(POVERTY_DATA_PATH)
        # 筛选芝加哥2019年的数据，并以CBG代码为索引
        income_df = income_df[(income_df['City Name'] == f'{CITY} city') & (income_df['Year'] == 2019)]
        income_df = income_df.set_index('CBG Code')
        print("Income data loaded and filtered for Chicago 2019.")
    except FileNotFoundError:
        print(f"Error: Income data file not found at {POVERTY_DATA_PATH}")
        return

    # --- 1. Initialize Map ---
    map_data_path = MAP_DATA_PATH.get(CITY)
    if not map_data_path:
        print(f"Error: Map data path for city '{CITY}' not found in constant.py")
        return

    try:
        chicago_map = Map(
            data_cache=map_data_path, 
            map_scope=MAP_SCOPE[CITY]
        )
        print("Map initialized successfully.")
    except Exception as e:
        print(f"Error initializing map: {e}")
        return

    # --- 2. Select a random subset of CBGs (AOIs) ---
    all_aoi_keys = list(chicago_map.aois.keys())
    if len(all_aoi_keys) < num_cbgs_to_select:
        print(f"Warning: Total AOIs ({len(all_aoi_keys)}) is less than requested subset ({num_cbgs_to_select}). Using all available AOIs.")
        selected_aoi_keys = all_aoi_keys
    else:
        selected_aoi_keys = random.sample(all_aoi_keys, num_cbgs_to_select)
    
    print(f"Randomly selected {len(selected_aoi_keys)} CBGs for agent generation.")

    # --- 3. Generate Agents in selected CBGs ---
    all_agents_df = pd.DataFrame()
    total_population_in_subset = sum(chicago_map.aois[key]['data']['Total population'].iloc[0] for key in selected_aoi_keys)
    
    # Add a small epsilon to avoid division by zero if total population is zero
    EPSILON = 1e-8

    for aoi_key in selected_aoi_keys:
        aoi_data = chicago_map.aois[aoi_key]['data'].iloc[0]
        
        # Calculate number of agents for this AOI based on its population proportion within the subset
        aoi_population = aoi_data['Total population']
        if total_population_in_subset > 0:
            proportion = aoi_population / total_population_in_subset
            num_agents_in_aoi = int(proportion * TOTAL_AGENTS_TO_GENERATE)
        else:
            num_agents_in_aoi = 0

        if num_agents_in_aoi <= 0:
            continue
            
        print(f"  - Generating {num_agents_in_aoi} agents in CBG: {aoi_key}")

        # Define demographic distributions for this AOI
        # Gender Distribution
        female_ratio = aoi_data['female_ratio']
        gender_probs = np.clip([female_ratio, 1 - female_ratio], 0, 1)
        gender_probs /= (gender_probs.sum() + EPSILON)
        
        # Race Distribution
        race_ratios = np.array([
            aoi_data['white_ratio'],
            aoi_data['asian_ratio'],
            aoi_data['black_ratio'],
            max(0.0, 1 - aoi_data['white_ratio'] - aoi_data['asian_ratio'] - aoi_data['black_ratio'])
        ])
        race_probs = np.clip(race_ratios, 0, 1)
        race_probs /= (race_probs.sum() + EPSILON)
        RACE_DIST = {'White': race_probs[0], 'Asian': race_probs[1], 'Black': race_probs[2], 'Others': race_probs[3]}

        # 获取该CBG的收入中位数
        try:
            median_income = income_df.loc[int(aoi_key)]['Median Household Income']
        except KeyError:
            print(f"  - Warning: No income data for CBG {aoi_key}. Using default income distribution.")
            # 如果找不到该CBG的收入数据，则使用一个默认的平均分布
            INCOME_DIST = {'Low': 0.2, 'Lower_Middle': 0.3, 'Middle': 0.35, 'Upper_Middle': 0.12, 'High': 0.03}
        else:
            INCOME_DIST = get_income_distribution_from_median(median_income)

        # Generate agents
        genders = np.random.choice(list(['Female', 'Male']), num_agents_in_aoi, p=list(gender_probs))
        races = np.random.choice(list(RACE_DIST.keys()), num_agents_in_aoi, p=list(RACE_DIST.values()))
        agent_types = np.random.choice(['egoist', 'altruist'], num_agents_in_aoi, p=[1 - altruist_ratio, altruist_ratio])
        residences = [aoi_key] * num_agents_in_aoi
        incomes = np.random.choice(list(INCOME_DIST.keys()), num_agents_in_aoi, p=list(INCOME_DIST.values()))

        agents_df = pd.DataFrame({
            "gender": genders,
            "race": races,
            "agent_type": agent_types,
            "residence": residences,
            "income_level": incomes
        })
        all_agents_df = pd.concat([all_agents_df, agents_df], ignore_index=True)

    # --- 4. Save to JSON ---
    output_path = os.path.join(os.path.dirname(__file__), "citizens.json")
    all_agents_df.to_json(output_path, orient="records", indent=2)
    
    print(f"\n--- Generation Complete ---")
    print(f"Total agents generated: {len(all_agents_df)}")
    print(f"Agent data saved to: {output_path}")

if __name__ == "__main__":
    main() 