import os
import shutil
import yaml
import random
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging
import argparse

from src.environment.map import Map
from utils import LLM
from src.agents.resident import ResidentAgent
from constant import MAP_SCOPE, MAP_DATA_PATH

RESULTS_P = f"results_segregation_{datetime.now().strftime('%Y%m%d')}/"

class SegregationSimulation:
    """基于收入和种族的谢林模型居民隔离模拟"""
    def __init__(self, config_path: str = "config.yaml"):
        try:
            # 加载配置
            if not Path(config_path).exists():
                raise FileNotFoundError(f"配置文件未找到于: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            city = self.config['environment']['city']
            map_data_path = MAP_DATA_PATH.get(city)
            if not map_data_path:
                raise ValueError(f"Map data path for city '{city}' not found in constant.py")

            self.map = Map(
                data_cache=map_data_path,
                map_scope=MAP_SCOPE[city]
            )
            
            # 加载CBG数据
            self.cbg_data = self._load_cbg_data('dataset/01_Poverty_CBG_level.csv')

            # 加载街景数据
            self.street_view_data = self._load_street_view_data('cache/cbg_meta_img_sampled_summary.json')

            # 初始化模型
            self.llm = LLM(
                model_name=self.config['llm']['model_name'], 
                platform=self.config['llm']['platform'], 
                api_key=self.config['llm']['api_key']
            )
            
            # 初始化agent列表
            self.residents: List[ResidentAgent] = []
            
            # 初始化模拟步骤参数
            self.total_steps = self.config['simulation']['total_steps']
            self.current_step = 0
            
            # 创建结果目录
            self.results_path = Path(RESULTS_P)
            self.results_path.mkdir(exist_ok=True)
            self.trajectories_path = self.results_path / 'trajectories'
            self.trajectories_path.mkdir(exist_ok=True)

            # --- 设置日志 ---
            log_path = self.results_path / 'migration_log.txt'
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - STEP %(levelname)s - %(message)s',
                filename=log_path,
                filemode='w'
            )
            self.logger = logging.getLogger(__name__)

        except FileNotFoundError as e:
            print(f"错误: 所需文件未找到: {e.filename}")
            raise
        except yaml.YAMLError as e:
            print(f"读取配置文件时出错: {str(e)}")
            raise
        except Exception as e:
            print(f"初始化错误: {str(e)}")
            raise

    def _load_cbg_data(self, poverty_data_path: str) -> pd.DataFrame:
        """加载并处理CBG贫困数据。"""
        df = pd.read_csv(poverty_data_path)
        # 筛选芝加哥市2019年数据
        df_chicago_2019 = df[(df['City Name'] == 'Chicago city') & (df['Year'] == 2019)]
        
        if df_chicago_2019.empty:
            print("警告: 未找到2019年芝加哥市的CBG数据，请检查数据文件。")
            return pd.DataFrame()

        df_2019 = df_chicago_2019.set_index('CBG Code')
        return df_2019[['Median Household Income']]

    def _load_street_view_data(self, street_view_data_path: str) -> Dict:
        """加载街景摘要数据。"""
        try:
            with open(street_view_data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 街景数据文件未找到于: {street_view_data_path}. 环境描述将不可用。")
            return {}
        except json.JSONDecodeError:
            print(f"警告: 解析街景数据文件时出错: {street_view_data_path}. 环境描述将不可用。")
            return {}

    def initialize_agents(self):
        """初始化所有居民Agent"""
        with open("./agent_initialization/citizens.json", "r", encoding="utf-8") as f:
            citizens_data = json.load(f)
        
        for i, citizen_data in enumerate(citizens_data):
            agent_id = 'R' + str(i)
            residence_cbg = str(citizen_data["residence"])
            
            if int(residence_cbg) not in self.cbg_data.index:
                print(f"警告: Agent {agent_id} 的居住CBG {residence_cbg} 在CBG数据中未找到。跳过此agent。")
                continue

            resident = ResidentAgent(
                agent_id=agent_id,
                gender=citizen_data["gender"],
                race=citizen_data["race"],
                agent_type=citizen_data.get("agent_type", "egoist"), # 默认为egoist
                residence=residence_cbg,
                income_level=citizen_data["income_level"],
                current_location=residence_cbg,
                historical_trajectory=[(0, residence_cbg)],
                visited_locations={residence_cbg: 1}
            )
            self.residents.append(resident)
            
            # 为每个Agent创建初始轨迹文件
            self._initialize_trajectory_file(resident)
    
    def _initialize_trajectory_file(self, agent: ResidentAgent):
        """为每个agent创建并初始化轨迹日志文件。"""
        trajectory_file = self.trajectories_path / f"{agent.agent_id}.json"
        initial_data = {
            "agent_profile": agent.get_attributes(),
            "initial_residence": agent.residence,
            "migration_history": []
        }
        with open(trajectory_file, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=4)

    def _get_social_utility(self, cbg_code: str, agent: ResidentAgent) -> float:
        """根据种族和收入相似性计算社会效用值。"""
        residents_in_cbg = [r for r in self.residents if r.current_location == cbg_code]
        total_pop = len(residents_in_cbg)
        if total_pop == 0:
            return 0.0

        # 1. 计算种族效用
        same_race_pop = sum(1 for r in residents_in_cbg if r.race == agent.race)
        rho_race = same_race_pop / total_pop
        utility_race = 2 * rho_race if rho_race <= 0.5 else 2 * (1 - rho_race)

        # 2. 计算收入效用
        same_income_pop = sum(1 for r in residents_in_cbg if r.income_level == agent.income_level)
        rho_income = same_income_pop / total_pop
        utility_income = 2 * rho_income if rho_income <= 0.5 else 2 * (1 - rho_income)

        # 3. 组合社会效用（平均值）
        return (utility_race + utility_income) / 2

    def _get_average_system_utility(self) -> float:
        """计算整个系统的平均社会效用"""
        if not self.residents:
            return 0.0
        
        total_utility = sum(self._get_social_utility(r.current_location, r) for r in self.residents)
        return total_utility / len(self.residents)

    def _get_cbg_demographics(self, cbg_code: str) -> Dict[str, Any]:
        """获取给定CBG的人口统计信息"""
        cbg_code_int = int(cbg_code)
        if cbg_code_int not in self.cbg_data.index:
            return {"error": "CBG not found"}
            
        income_info = self.cbg_data.loc[cbg_code_int].to_dict()
        
        residents_in_cbg = [r for r in self.residents if r.current_location == cbg_code]
        racial_counts = {}
        for r in residents_in_cbg:
            racial_counts[r.race] = racial_counts.get(r.race, 0) + 1
        
        total_pop = len(residents_in_cbg)
        racial_composition = {race: count / total_pop for race, count in racial_counts.items()} if total_pop > 0 else {}

        return {
            "median_income": income_info.get("Median Household Income"),
            "racial_composition": racial_composition,
            "population": total_pop
        }

    def _run_migration_step(self):
        """为所有agent运行一步迁移模拟"""
        # 创建一个副本以避免在迭代时修改列表
        agents_to_process = list(self.residents)
        
        # 依次处理每个agent
        for agent in tqdm(agents_to_process, desc=f"Step {self.current_step} Migration"):
            self._decide_and_move(agent)

    def _decide_and_move(self, resident: ResidentAgent):
        """为单个agent决定是否移动并执行"""
        current_cbg_code = resident.current_location
        
        # 获取所有当前有居民居住的CBG的唯一列表
        populated_cbg_codes = set(r.current_location for r in self.residents)
        
        # 备选地点是除了自己当前位置之外的所有已居住CBG
        potential_cbg_codes = [cbg for cbg in populated_cbg_codes if cbg != current_cbg_code]

        # 如果没有其他社区可选，则直接跳过
        if not potential_cbg_codes:
            self._update_trajectory_file(resident, current_cbg_code, current_cbg_code, False, "No other populated CBGs to consider.", {})
            return

        # 限制备选地点数量，最多50个（随机选择）
        MAX_OPTIONS = 50
        if len(potential_cbg_codes) > MAX_OPTIONS:
            potential_cbg_codes = random.sample(potential_cbg_codes, MAX_OPTIONS)
        
        # 获取社会效用和环境描述
        options = {}
        for cbg in potential_cbg_codes:
            social_utility = self._get_social_utility(cbg, resident)
            env_summary = self.street_view_data.get(cbg, {}).get("summary", "No street view data available.")
            options[cbg] = {
                "social_utility": social_utility,
                "summary": env_summary
            }
        
        current_social_util = self._get_social_utility(current_cbg_code, resident)
        options[current_cbg_code] = {
            "social_utility": current_social_util,
            "summary": self.street_view_data.get(current_cbg_code, {}).get("summary", "No street view data available.")
        }
    
        # 获取平均系统效用（主要给利他主义者参考）
        avg_system_utility = self._get_average_system_utility()
    
        prompt = self._construct_migration_prompt(
            resident, 
            options,
            avg_system_utility
        )
        
        # 日志记录Prompt
        self.logger.info(f"--- Agent: {resident.agent_id} (Type: {resident.agent_type}) ---")
        self.logger.info(f"Current Location: {current_cbg_code}, Social Utility: {options[current_cbg_code]['social_utility']:.4f}")
        self.logger.info("[PROMPT SENT TO LLM]")
        self.logger.info(prompt)
        
        response = self.llm.generate(prompt)
        # 解析Json
        try:
            decision_data = json.loads(response.strip())
        except json.JSONDecodeError:
            try:
                response_text = response.strip()
                json_start = response_text.find('{')
                json_end = response_text.rfind('}')
                
                if json_start != -1 and json_end != -1 and json_start < json_end:
                    json_str = response_text[json_start:json_end+1]
                    decision_data = json.loads(json_str)
                else:
                    decision_data = None
            except json.JSONDecodeError:
                decision_data = None

        if decision_data:
            new_loc_cbg = decision_data.get("move_to_cbg")
            thinking_process = decision_data.get("thinking")
        else:
            # 解析失败
            new_loc_cbg = response.strip()
            thinking_process = "LLM did not return valid JSON. Raw response used."

        # 日志记录
        self.logger.info(f"[LLM RESPONSE RECEIVED]: {response.strip()}")
        self.logger.info("-" * 40 + "\n")

        moved = False
        if new_loc_cbg in options and new_loc_cbg != current_cbg_code:
            resident.current_location = new_loc_cbg
            moved = True
        
        # 更新轨迹文件
        self._update_trajectory_file(resident, current_cbg_code, new_loc_cbg, moved, thinking_process, options)

    def _construct_migration_prompt(self, resident: ResidentAgent, options: Dict[str, dict], avg_system_utility: float) -> str:
        """基于社会效用和环境描述的移动Prompt"""
        
        egoist_rules = """
As an 'egoist', your goal is to find the best possible neighborhood for yourself by balancing two key factors:
1.  'Social Utility': A quantitative score of your potential social comfort. Higher is generally better.
2.  'Environment Summary': A qualitative description of the area's physical character and safety.
Your task is to weigh both the numbers and the text description to choose the location that offers the best overall quality of life for you. A place with the highest utility might not be your best choice if its environment is poor, and a great environment might not compensate for a very low social utility.
"""
        altruist_rules = f"""
As an 'altruist', your primary goal is to maximize the 'Average System Social Utility'. The current average is {avg_system_utility:.4f}. You should choose the action (move or stay) that you believe will best increase this average. You can use the 'Environment Summary' as context for your decision, but the system's overall social well-being is your priority.
"""

        prompt = f"""You are a resident of Chicago participating in a socio-environmental simulation. Your goal is to decide whether to move to a new neighborhood. Your decision will be based on a quantitative 'Social Utility' score and a qualitative 'Environment Summary'.

Your Profile:
- Race: {resident.race}
- Income Level: {resident.income_level}
- Agent Type: {resident.agent_type}

Decision Rules:
Your decision must be based on your 'Agent Type'.
{''.join(egoist_rules if resident.agent_type == 'egoist' else altruist_rules)}

Metrics Explained:
1.  'Social Utility': A score from 0 to 1 based on the similarity of your neighbors (both race and income level). A higher score means more social comfort. This is your primary quantitative metric.
2.  'Environment Summary': A text description of the neighborhood's physical environment, based on street-level images. This provides qualitative context.

Your Options:
Here are your potential neighborhoods, including your current one.

"""
        for cbg, utils in options.items():
            prompt += f"""- Neighborhood (CBG: {cbg}):
  - Social Utility: {utils['social_utility']:.4f}
  - Environment Summary: "{utils['summary']}"
"""

        prompt += """
Your Task:
Based on your profile, your agent type's rules, the quantitative Social Utility, and the qualitative Environment Summary, make a decision. Your response MUST be a JSON object with two keys:
1. "move_to_cbg": The CBG code of the neighborhood you choose to live in (this can be your current one if you decide not to move).
2. "thinking": A brief, one-sentence explanation for your decision, reflecting your agent type's logic and how you weighed the social utility against the environmental description.

Example for an egoist:
{
  "move_to_cbg": "17031842400",
  "thinking": "As an egoist, I am moving to the neighborhood with the highest Social Utility, as the environment summary also seems acceptable."
}

Example for an altruist:
{
  "move_to_cbg": "17031842500",
  "thinking": "As an altruist, I will stay in my current location. Although another option has higher personal utility, moving might disrupt the social balance and lower the overall system utility."
}

Your decision (JSON format only):
"""
        return prompt

    def _update_trajectory_file(self, agent: ResidentAgent, from_cbg: str, to_cbg: str, moved: bool, thinking: str, options: dict):
        """轨迹文件记录信息"""
        trajectory_file = self.trajectories_path / f"{agent.agent_id}.json"
        
        simplified_options = {
            cbg: {
                "social_utility": utils["social_utility"]
            } for cbg, utils in options.items()
        }

        step_data = {
            "step": self.current_step,
            "decision": "move" if moved else "stay",
            "from_cbg": from_cbg,
            "to_cbg": to_cbg,
            "reasoning": thinking,
            "utility_context": {
                "options_considered": simplified_options,
                "agent_utility_before": options.get(from_cbg, {}).get('social_utility'),
                "agent_utility_after": options.get(to_cbg, {}).get('social_utility'),
            }
        }
        
        with open(trajectory_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["migration_history"].append(step_data)
            f.seek(0)
            f.truncate()
            json.dump(data, f, ensure_ascii=False, indent=4)

    def run_simulation(self):
        """主模拟循环"""
        print("开始基于Schelling模型的隔离模拟...")
        self.logger.info("Simulation Started")
        
        self.initialize_agents()
        print(f"已初始化 {len(self.residents)} 个agent。")
        self.logger.info(f"Initialized {len(self.residents)} agents.")
        
        for step in range(self.total_steps):
            self.current_step = step
            print(f"\n--- 步骤 {step+1}/{self.total_steps} ---")
            self.logger.info(f"Processing Step {self.current_step}")
            
            self._run_migration_step()
            
        print(f"模拟完成。结果保存在 '{self.results_path}' 目录中。")
        self.logger.info("Simulation Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行基于Agent的隔离模拟。")
    parser.add_argument('--steps', type=int, help="要运行的模拟总步数，将覆盖配置文件中的设置。")
    args = parser.parse_args()

    simulation = SegregationSimulation()

    if args.steps:
        print(f"检测到命令行参数：模拟将运行 {args.steps} 步。")
        simulation.total_steps = args.steps
    
    simulation.run_simulation()