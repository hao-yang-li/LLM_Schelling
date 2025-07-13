import numpy as np
from typing import Dict, List, Tuple, Any
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
import pyproj
from functools import lru_cache
import pickle
import random
import json

class Map:
    """地图类，用于管理GIS地图和CBG数据"""
    
    def __init__(self, data_cache: str, map_scope: dict):
        """
        初始化地图
        
        Args:
            data_cache: 地图数据缓存文件路径
            map_scope: 地图范围
        """
        self.map_scope = None
        if map_scope:
            left_bottom = map_scope[0]['left_bottom']
            right_top = map_scope[0]['right_top']
            self.map_scope = box(left_bottom[0], left_bottom[1], right_top[0], right_top[1])
        
        # 加载地图数据
        map_data = self._load_map_data(data_cache)
        map_data = self._map_preprocess(map_data)
        cbgs = map_data["cbgs"]
        with open('./cache/cbg_meta_img_sampled_summary.json', 'r') as f:
            data = json.load(f)
        cbgs_renewed = {}
        for k, v in cbgs.items():
            if k in data.keys() and len(data[k]['images']) > 3:
                cbgs_renewed[k] = v
        self.aois: Dict[str, dict] = cbgs_renewed

        self.pois: Dict[str, dict] = {poi_id: poi for poi_id, poi in map_data["pois"].items() if poi['poi_cbg'] in cbgs_renewed.keys()}
        
        # 创建投影转换器
        self.proj = pyproj.Proj(proj='utm', zone=16, ellps='WGS84')

    def _load_map_data(self, data_cache: str) -> Dict:
        """
        从缓存文件加载地图数据
        
        Args:
            data_cache: 地图数据缓存文件路径
            
        Returns:
            Dict: 地图数据
        """
        with open(data_cache, 'rb') as f:
            return pickle.load(f)

    def _map_preprocess(self, map_data):
        for poi in map_data["pois"].values():
            poi['poi_cbg'] = str(poi['poi_cbg'])
            self.proj = pyproj.Proj(proj='utm', zone=16, ellps='WGS84')
            poi["shapely_xy"] = Point(self.proj(poi["lnglat"][0], poi["lnglat"][1]))
            poi["shapely_lnglat"] = Point(poi["lnglat"])
            # 处理poi 中的normalized_visits_by_state_scaling, 将nan 设置为0
            poi["normalized_visits_by_state_scaling"] = np.nan_to_num(poi["normalized_visits_by_state_scaling"])
            # 处理poi 中的visitor_home_cbgs, 将key转换为str
            visitor_home_cbgs = {}
            for k, v in poi["visitor_home_cbgs"].items():
                try:
                    visitor_home_cbgs[str(int(k))] = v
                except Exception as err:
                    visitor_home_cbgs[str(k)] = v
            poi["visitor_home_cbgs"] = visitor_home_cbgs

        for aoi in map_data["cbgs"].values():
            try:
                aoi["shapely_lnglat"] = Polygon(list(aoi['geometry'].iloc[0].geoms)[0])
                l = list(list(aoi['geometry'].iloc[0].geoms)[0].exterior.coords)
                l = [self.proj(x,y) for (x,y) in l]
                aoi["shapely_xy"] = Polygon(l)
            except Exception as err:
                # print(aoi)
                aoi["shapely_lnglat"] = Polygon([])
                aoi["shapely_xy"] = Polygon([])
                # print(aoi['geometry'], type(aoi['geometry']))

        # 将map_data['cbgs] 的key 转换为str
        map_data['cbgs'] = {str(k): v for k, v in map_data['cbgs'].items()}

        # 保留在范围内的POI和AOI
        if self.map_scope is None:
            return map_data

        aois = {}
        for aoi_id, aoi in map_data["cbgs"].items():
            #if aoi["shapely_lnglat"].intersects(self.map_scope):
            aois[str(aoi_id)] = aoi
        
        pois = {}
        for poi_id, poi in map_data["pois"].items():
            # 保留在范围内的POI
            #if poi["shapely_lnglat"].intersects(self.map_scope):
            poi['poi_cbg'] = str(poi['poi_cbg'])
            pois[poi_id] = poi

        for aoi_id, aoi in map_data['cbgs'].items():
            if str(aoi_id) not in aois:
                continue
            pois_in_aoi = []
            for poi_id in aois[str(aoi_id)]['poi']:
                if poi_id in pois:
                    pois_in_aoi.append(poi_id)

            aoi['poi'] = pois_in_aoi
            if len(pois_in_aoi) == 0: 
                del aois[str(aoi_id)]

        map_data["cbgs"] = aois
        map_data["pois"] = pois

        return map_data
    def get_aoi(self, aoi_id: str) -> Dict[str, Any]:
        """
        获取CBG信息
        
        Args:
            aoi_id: CBG ID
            
        Returns:
            Dict[str, Any]: CBG信息
        """
        return self.aois.get(aoi_id, {})

    @lru_cache(maxsize=1024)
    def calculate_distance(self, cbg_id1: str, cbg_id2: str) -> float:
        """
        计算两个CBG之间的距离
        
        Args:
            cbg_id1: 第一个CBG的ID
            cbg_id2: 第二个CBG的ID
            
        Returns:
            float: 距离（米）
        """
        aoi1 = self.get_aoi(cbg_id1)
        aoi2 = self.get_aoi(cbg_id2)

        if not aoi1 or not aoi2:
            return float('inf')
            
        # 使用CBG质心计算距离
        centroid1 = aoi1['shapely_xy'].centroid
        centroid2 = aoi2['shapely_xy'].centroid
        
        return centroid1.distance(centroid2)
    
    def get_cbg(self, lat: float, lng: float) -> str:
        """
        根据经纬度获取CBG ID
        
        Args:
            lat: 纬度
            lng: 经度
            
        Returns:
            str: CBG ID
        """
        point = Point(lng, lat)
        nearest_distance = float('inf')
        nearest_cbg_id = None

        for poi_id, poi in self.pois.items():
            dist = point.distance(poi['shapely_lnglat'])
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_cbg_id = poi['poi_cbg']
                
        return nearest_cbg_id if nearest_cbg_id else None