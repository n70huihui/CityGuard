import numpy as np
import matplotlib.pyplot as plt
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import random

from tqdm.contrib.discord import tdrange


def latlon_to_grid(lat: float, lon: float,
                   map_width: int, map_height: int,
                   lat_range: tuple[float, float] = (-90, 90),
                   lon_range: tuple[float, float] = (-180, 180)) -> tuple[int, int]:
    """
    将经纬度坐标转换为地图网格坐标
    :param lat: 纬度 [-90, 90]
    :param lon: 经度 [-180, 180]
    :param map_width: 地图宽度 (网格列数)
    :param map_height: 地图高度 (网格行数)
    :param lat_range: 纬度范围 (min, max)
    :param lon_range: 经度范围 (min, max)
    :return: (x, y) 地图网格坐标
    """
    # 确保输入在有效范围内
    lat = max(lat_range[0], min(lat_range[1], lat))
    lon = max(lon_range[0], min(lon_range[1], lon))

    # 计算归一化坐标 (0-1)
    lat_norm = (lat - lat_range[0]) / (lat_range[1] - lat_range[0])
    lon_norm = (lon - lon_range[0]) / (lon_range[1] - lon_range[0])

    # 转换为网格坐标
    # 注意: 地图网格的y轴方向与纬度方向相反
    x = int(lon_norm * (map_width - 1))
    y = int((1 - lat_norm) * (map_height - 1))  # 翻转y轴

    # 确保坐标在网格范围内
    x = max(0, min(map_width - 1, x))
    y = max(0, min(map_height - 1, y))

    return x, y


def get_quadrant(current_x: int, current_y: int,
                 target_x: int, target_y: int) -> int:
    """
    根据车辆坐标和目标点坐标确定所在象限
    :param current_x: 车辆 X 坐标
    :param current_y: 车辆 Y 坐标
    :param target_x: 目标点 X 坐标
    :param target_y: 目标点 Y 坐标
    :return: 象限编号 (0-3)
    1 | 2
    --+--
    0 | 3
    """
    if current_x >= target_x and current_y < target_y:
        return 0
    elif current_x < target_x and current_y <= target_y:
        return 1
    elif current_x <= target_x and current_y > target_y:
        return 2
    else:
        return 3


class MapSimulator:
    """
    地图模拟器，支持车辆管理
    """
    def __init__(self, width=50, height=50):
        """
        初始化地图模拟器
        :param width: 地图宽度
        :param height: 地图高度
        """
        self.width = width
        self.height = height
        self.target_point = None
        # 创建空白地图 (1表示可通行，0表示障碍)
        self.grid_matrix = np.ones((height, width), dtype=np.int32)

        # 车辆位置集合
        self.vehicle_positions = set()

        # 添加随机障碍物
        self._add_random_obstacles(obstacle_density=0.2)

        # 添加随机拥堵区域
        self._add_congestion_zones()

        # 创建路径规划网格
        self.path_grid = Grid(matrix=self.grid_matrix)

    def get_map_grid(self) -> np.ndarray:
        """
        获取地图网格
        :return: 当前地图的网格矩阵
        """
        return self.grid_matrix

    def add_vehicle(self, positions: list[tuple[int, int]]) -> None:
        """
        添加车辆到地图
        :param positions: 车辆位置列表 [(x1, y1), (x2, y2), ...]
        """
        for pos in positions:
            x, y = pos
            # 确保位置在地图范围内
            if 0 <= x < self.height and 0 <= y < self.width:
                # 当前位置设置为车辆
                self.grid_matrix[x][y] = 1
                self.vehicle_positions.add(pos)

    def remove_vehicle(self, positions: list[tuple[int, int]]) -> None:
        """
        从地图移除车辆
        :param positions: 车辆位置列表 [(x1, y1), (x2, y2), ...]
        """
        for pos in positions:
            if pos in self.vehicle_positions:
                self.vehicle_positions.remove(pos)

    def _add_random_obstacles(self, obstacle_density: float=0.1) -> None:
        """
        添加随机障碍物
        """
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < obstacle_density:
                    self.grid_matrix[y][x] = 0  # 0 表示障碍物

    def _add_congestion_zones(self) -> None:
        """
        添加拥堵区域（权重较高的区域）
        """
        for _ in range(5):  # 添加5个拥堵区域
            center_x = random.randint(5, self.width - 5)
            center_y = random.randint(5, self.height - 5)
            radius = random.randint(1, 3)

            # 在圆形区域内增加权重
            for y in range(max(0, center_y - radius), min(self.height, center_y + radius)):
                for x in range(max(0, center_x - radius), min(self.width, center_x + radius)):
                    # 计算点到中心的距离
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist <= radius:
                        # 中心区域权重最高，向外递减
                        weight = max(1, int(10 * (1 - dist / radius)))
                        self.grid_matrix[y][x] = weight

    def get_nearby_vehicle_positions(self, radius: int) -> set[tuple[int, int]]:
        """
        获取附近车辆位置
        :param radius: 搜索半径
        :return: 车辆位置信息
        """
        positions = set()
        for x, y in self.vehicle_positions:
            distance = np.sqrt((x - self.target_point[0]) ** 2 + (y - self.target_point[1]) ** 2)
            if distance < radius:
                positions.add((x, y))
        return positions

    def add_target_point(self, target: tuple[int, int]) -> None:
        """
        添加目标点
        """
        self.target_point = target

    def find_path(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """
        使用A*算法查找路径
        :param start: 起点坐标 (x, y)
        :param end: 终点坐标 (x, y)
        :return: 路径列表 [(x1, y1), (x2, y2), ...]
        """
        # 设置起点和终点
        start_node = self.path_grid.node(start[0], start[1])
        end_node = self.path_grid.node(end[0], end[1])

        # 创建A*查找器
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

        # 查找路径
        path, _ = finder.find_path(start_node, end_node, self.path_grid)
        return path

    def visualize(self, vehicle_paths: dict=None) -> None:
        """
        可视化地图和车辆路径
        :param vehicle_paths: 车辆路径字典 {vehicle_id: 路径}
        """
        # 创建可视化网格
        vis_grid = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 填充网格颜色 - 简化的地图表示
        for y in range(self.height):
            for x in range(self.width):
                value = self.grid_matrix[y][x]
                if value == 0:  # 障碍物 - 黑色
                    vis_grid[y][x] = [0, 0, 0]
                elif value == 1:  # 普通道路 - 浅灰色
                    vis_grid[y][x] = [200, 200, 200]
                else:  # 拥堵区域 - 红色（根据权重调整深浅）
                    # 安全计算颜色强度，避免溢出
                    intensity = 100 + value * 15
                    intensity = min(intensity, 255)
                    vis_grid[y][x] = [intensity, 0, 0]

        # 绘制路径
        if vehicle_paths:
            for path in vehicle_paths.values():
                if path:
                    for px, py in path:
                        # 浅绿色路径 [100, 255, 100]
                        vis_grid[px][py] = [100, 255, 100]

        # 绘制车辆位置
        for (x, y) in self.vehicle_positions:
            vis_grid[x][y] = [0, 0, 255]  # 蓝色车辆

        # 绘制目标 - 简单的黄色方块
        if self.target_point:
            x, y = self.target_point
            vis_grid[x][y] = [255, 255, 0]  # 黄色目标

        # 显示图像
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_grid)
        plt.title('CityGuard')
        plt.axis('off')
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建地图模拟器
    simulator = MapSimulator(width=30, height=30)

    # 随机生成目标点
    end = (random.randint(0, 29), random.randint(0, 29))
    while simulator.grid_matrix[end[0]][end[1]] == 0:
        end = (random.randint(0, 29), random.randint(0, 29))
    simulator.target_point = end

    # 添加5辆车
    vehicles = []
    for _ in range(1):
        while True:
            x, y = random.randint(0, 29), random.randint(0, 29)
            if simulator.grid_matrix[x][y] != 0:  # 确保不在障碍物上
                vehicles.append((x, y))
                break

    # 将车辆添加到地图
    simulator.add_vehicle(vehicles)

    # 随机选择3辆车计算路径
    selected_vehicles = random.sample(vehicles, 1)
    vehicle_paths = {}

    print("=== 车辆路径规划 ===")
    for i, position in enumerate(selected_vehicles):
        print(get_quadrant(position[0], position[1], simulator.target_point[0], simulator.target_point[1]))
        vehicle_id = f"Vehicle-{i + 1}"
        path = simulator.find_path(position, end)
        vehicle_paths[vehicle_id] = path

    # 可视化结果
    simulator.visualize(vehicle_paths=vehicle_paths)

    # 移除所有车辆
    simulator.remove_vehicle(vehicles)
    print("车辆已移除")
