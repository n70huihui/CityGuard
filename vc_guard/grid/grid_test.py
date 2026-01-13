import json
import random

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple

from vc_guard.edge.vechicle import AgentCard
from env_utils.llm_args import *
from vc_guard.globals.vehicles import vehicles

random.seed(42)

class MapCell(BaseModel):
    """地图单元格表示"""
    cell_type: str  # 'road', 'building', 'park', 'obstacle'
    congestion: float  # 0-1, 拥堵程度
    elevation: float  # 海拔高度

class MapGrid(BaseModel):
    """网格化地图表示"""
    width: int
    height: int
    cells: Dict[Tuple[int, int], MapCell]  # (x,y) -> 单元格
    landmarks: Dict[str, Tuple[int, int]]  # 地标名称 -> 位置

class VehicleDecisionInput(BaseModel):
    """大模型决策输入数据"""
    task_location: Tuple[float, float]
    task_description: str
    map_grid: MapGrid
    vehicles: List['AgentCard']  # 使用前向引用

class VehicleDecisionOutput(BaseModel):
    """大模型决策输出数据"""
    selected_vehicles: List[str] = Field(description="选择的车辆ID")
    vehicle_paths: Dict[str, List[Tuple[int, int]]] = Field(description="每辆车的路径坐标")
    reasoning: str = Field(description="决策理由")

# 新增提示词模板
llm_planner_template = """
你是一个智能交通规划系统。请基于以下信息做出最优决策：

## 任务描述
{task_description}
目标位置坐标: ({target_x}, {target_y})

## 地图环境
地图尺寸: {map_width}x{map_height}
关键地标: 
{landmarks}
拥堵区域: 
{congestion_areas}

## 可用车辆
{vehicle_details}
"""


def _get_agent_cards() -> list[str]:

    car_id_set: set[str] = set([vehicle.car_id for vehicle in vehicles])

    # 线程池并行处理
    from vc_guard.globals.executor import vehicle_executor
    agent_cards: list[str] = vehicle_executor.execute_tasks(
        vehicle_list=vehicles,
        best_vehicle_id_set=car_id_set,
        method_name='get_agent_card',
        return_results=True
    )

    return agent_cards

def _create_simulated_map(width=100, height=100):
    """创建模拟地图"""
    map_grid = MapGrid(width=width, height=height, cells={}, landmarks={})

    # 填充基本道路
    for x in range(width):
        for y in range(height):
            cell_type = "road"
            congestion = random.uniform(0, 0.3)

            # 随机添加地标和障碍
            if random.random() < 0.02:
                cell_type = "building"
                congestion = min(1.0, congestion + 0.7)
            elif random.random() < 0.01:
                cell_type = "park"
                congestion = max(0, congestion - 0.2)
            elif random.random() < 0.01:
                cell_type = "obstacle"

            map_grid.cells[(x, y)] = MapCell(
                cell_type=cell_type,
                congestion=congestion,
                elevation=random.uniform(0, 10)
            )

    # 添加关键地标
    map_grid.landmarks = {
        "市中心广场": (width // 2, height // 2),
        "交通枢纽": (width // 4, height // 4),
        "科技园区": (3 * width // 4, 3 * height // 4)
    }

    return map_grid

def _llm_based_planner(
                       map_grid: MapGrid,
                       task_description: str,
                       agent_cards: List[str],
                       num_vehicles: int) -> VehicleDecisionOutput:
    """使用LLM进行智能规划"""
    # 将经纬度转换为网格坐标
    target_x = 8
    target_y = 8

    # 准备提示词
    landmarks_str = "\n".join(
        f"- {name}: ({x}, {y})"
        for name, (x, y) in map_grid.landmarks.items()
    )

    congestion_areas = []
    for (x, y), cell in map_grid.cells.items():
        if cell.congestion > 0.6:
            congestion_areas.append(f"- 区域({x},{y}): 拥堵程度 {cell.congestion:.2f}")

    vehicle_details = []
    for card_json in agent_cards:
        card = AgentCard(**json.loads(card_json))
        x = int((card.location[0] + 90) / 180 * map_grid.width)
        y = int((card.location[1] + 180) / 360 * map_grid.height)
        vehicle_details.append(
            f"- 车辆 {card.car_id}: 位置({x},{y}), "
            f"速度{card.speed}, "
            f"状态{'工作中' if card.is_working else '空闲'}"
        )

    prompt = llm_planner_template.format(
        task_description=task_description,
        target_x=target_x,
        target_y=target_y,
        map_width=map_grid.width,
        map_height=map_grid.height,
        landmarks=landmarks_str,
        congestion_areas="\n".join(congestion_areas),
        vehicle_details="\n".join(vehicle_details),
        num_vehicles=num_vehicles
    )

    # 调用LLM进行规划
    agent = create_agent(
        model=ChatOpenAI(base_url=base_url, model=model, api_key=api_key),
        tools=[],
        response_format=ToolStrategy(VehicleDecisionOutput)
    )
    response = agent.invoke({"messages": [prompt]})
    return response["structured_response"]

if __name__ == "__main__":

    # 创建模拟地图
    map_grid = _create_simulated_map(width=30, height=30)

    agent_cards = _get_agent_cards()

    decision = _llm_based_planner(map_grid=map_grid, task_description='查找城市地段异味根因', agent_cards=agent_cards, num_vehicles=3)

    # 简单的地图可视化
    grid = [['·' for _ in range(map_grid.width)] for _ in range(map_grid.height)]

    # 标记障碍物和拥堵区域
    for (x, y), cell in map_grid.cells.items():
        if cell.cell_type == "obstacle":
            grid[y][x] = 'X'
        elif cell.congestion > 0.7:
            grid[y][x] = str(min(9, int(cell.congestion * 10)))

    # 标记地标
    for name, (x, y) in map_grid.landmarks.items():
        grid[y][x] = '★'

    # 标记任务目标
    target_x, target_y = 8, 8
    grid[target_y][target_x] = '◎'

    # 标记车辆和路径
    for vehicle_id, path in decision.vehicle_paths.items():
        if path:
            start_x, start_y = path[0]
            grid[start_y][start_x] = '⇧'  # 起点

            for i in range(1, len(path) - 1):
                x, y = path[i]
                grid[y][x] = '*'  # 路径点

            end_x, end_y = path[-1]
            grid[end_y][end_x] = '⇩'  # 终点

    # 打印地图
    print("\n地图可视化:")
    for row in grid:
        print(''.join(row))
    print('=' * 70)