from datetime import datetime
import random
from typing import List, Dict, Any, Optional

class ResidentAgent:
    """居民Agent类，代表普通市民"""
    
    def __init__(
        self,
        agent_id: str,
        gender: str,
        race: str,
        agent_type: str, # 'egoist' or 'altruist'
        residence: str,
        income_level: int,
        current_location: Optional[str] = None,
        historical_trajectory: Optional[List[tuple]] = None,
        visited_locations: Optional[Dict[str, int]] = None
    ):
        """
        初始化居民Agent
        
        Args:
            agent_id: 唯一标识符
            gender: 性别 (M/F/O)
            race: 种族
            agent_type: Agent的类型 ('egoist' 或 'altruist')
            residence: 居住地AOI ID
            income_level: 收入水平 (1-5)
        """
        self.agent_id = agent_id
        self.gender = gender
        self.race = race
        self.agent_type = agent_type
        self.residence = residence
        self.income_level = income_level
        
        # 动态属性
        self.current_location = current_location
        self.historical_trajectory = historical_trajectory  # 轨迹记录，每项为(step, location)的元组
        self.visited_locations = visited_locations  # 访问过的位置及其访问次数  
        if self.current_location is None:
            self.current_location = self.residence
            self.historical_trajectory = [(0, self.residence)]  # 初始位置
            self.visited_locations = {self.residence: 2}
    
    def update_location(self, new_location: str, current_step: int) -> None:
        """
        更新Agent的位置
        
        Args:
            new_location: 新的位置AOI ID
            current_step: 当前步数
        """
        self.current_location = new_location
        self.historical_trajectory.append((current_step, new_location))
        
        # 更新访问历史
        if new_location in self.visited_locations:
            self.visited_locations[new_location] += 1
        else:
            self.visited_locations[new_location] = 1
            
    def get_current_location(self) -> str:
        """获取当前位置"""
        return self.current_location
    
    def get_historical_trajectory(self) -> List[tuple]:
        """获取历史轨迹"""
        return self.historical_trajectory
    
    def get_visited_locations(self) -> Dict[str, int]:
        """获取访问过的位置及其访问次数"""
        return self.visited_locations
        
    def is_at_home(self) -> bool:
        """判断是否在家"""
        return self.current_location == self.residence
    
    def is_at_work(self) -> bool:
        """判断是否在工作地点"""
        return self.current_location == self.work_location
    
    def get_attributes(self) -> Dict[str, Any]:
        """获取Agent的所有属性"""
        return {
            'agent_id': self.agent_id,
            'gender': self.gender,
            'race': self.race,
            'agent_type': self.agent_type,
            'residence': self.residence,
            'income_level': self.income_level,
            'current_location': self.current_location,
            'historical_trajectory': self.historical_trajectory,
            'visited_locations': self.visited_locations
        }