import numpy as np
import matplotlib.pyplot as plt
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import random

from vc_guard.common.models import HandleObservationVo, Task, DataUnit
from vc_guard.edge.vechicle import Vehicle
from vc_guard.globals.executor import vehicle_executor


class GridSimulator:
    def __init__(self, height: int = 30, width: int = 30, num_of_vehicles: int = 20):
        # 初始化基础数据
        self.height = height
        self.width = width
        self.target_point = (random.randint(0, height - 1), random.randint(0, width - 1))
        self.num_of_vehicles = num_of_vehicles
        self.vehicles_dict = {}
        self.observation_point_dict = {}

        # 创建空白地图（1 表示可通行，0 表示障碍）
        self.grid_matrix = [[1] * self.width for _ in range(self.height)]

        # 添加随机障碍物
        self._add_random_obstacles()

        # 添加随机观测点
        self._add_observation_point()

        # 初始化地图车辆信息
        self._add_vehicles(self.num_of_vehicles)

        # 初始化路径地图
        self.path_grid = Grid(matrix=self.grid_matrix)


    def _add_random_obstacles(self, obstacle_density: float=0.2) -> None:
        for x in range(self.height):
            for y in range(self.width):
                if random.random() < obstacle_density:
                    self.grid_matrix[x][y] = 0  # 0 表示障碍物

    def _add_observation_point(self, obstacle_density: float=0.02) -> None:
        # TODO 根据目标点距离来判断用哪个观测
        observation_lst = ['car1', 'car2', 'car3', 'car4']
        for x in range(self.height):
            for y in range(self.width):
                if random.random() < obstacle_density and (x, y) != self.target_point and self.grid_matrix[x][y] != 0:
                    self.observation_point_dict[(x, y)] = random.choice(observation_lst)

    def _add_vehicles(self, num_of_vehicles: int) -> None:
        vehicles_lst = [Vehicle(height=self.height, width=self.width) for _ in range(num_of_vehicles)]
        for vehicle in vehicles_lst:
            x, y = vehicle.map_location
            self.vehicles_dict[vehicle.car_id] = vehicle
            self.grid_matrix[x][y] = 1

    def get_vehicle_observation_report(self,
                                       task: Task,
                                       type_name: str,
                                       file_id: str,
                                       vehicle_id_list: list[str]
                                       ) -> list[HandleObservationVo]:
        args = []

        for vehicle_id in vehicle_id_list:
            # 获取车辆位置
            vehicle_location = self.vehicles_dict[vehicle_id].map_location

            # 初始化最小距离和最近观测点
            min_distance = float('inf')
            nearest_obs_point = None

            # 遍历所有观测点计算距离
            for obs_point in self.observation_point_dict.keys():
                # 计算欧几里得距离
                distance = ((vehicle_location[0] - obs_point[0]) ** 2 +
                            (vehicle_location[1] - obs_point[1]) ** 2) ** 0.5

                # 更新最近观测点
                if distance < min_distance:
                    min_distance = distance
                    nearest_obs_point = obs_point

            # 获取观测点对应的值
            if nearest_obs_point:
                file_name = self.observation_point_dict[nearest_obs_point]
                args.append((task, DataUnit(type_name=type_name, file_id=file_id, file_name=file_name)))

        results = vehicle_executor.execute_tasks_different_args(
            vehicle_id_list=vehicle_id_list,
            vehicles=self.vehicles_dict,
            method_name='execute_task',
            args=args,
            return_results=True
        )

        return results

    def find_path(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        # 关键修改：交换坐标顺序
        start_node = self.path_grid.node(start[1], start[0])  # 交换顺序
        end_node = self.path_grid.node(end[1], end[0])  # 交换顺序

        # 创建A*查找器
        finder = AStarFinder()

        # 查找路径
        path, _ = finder.find_path(start_node, end_node, self.path_grid)

        return path

    def vehicle_id_find_path(self, vehicle_id: str, end: tuple[int, int]) -> list[tuple[int, int]]:
        start = self.vehicles_dict[vehicle_id].map_location
        return self.find_path(start, end)

    def visualize(self, vehicle_paths: dict = None) -> None:
        # 创建可视化网格
        vis_grid = np.zeros((self.height, self.width, 3), dtype=np.int16)

        # 先填充基础地形
        for x in range(self.height):
            for y in range(self.width):
                # 障碍物涂黑色，正常道路涂灰色
                vis_grid[x, y] = [0, 0, 0] if self.grid_matrix[x][y] == 0 else [200, 200, 200]

        # 绘制路径
        if vehicle_paths:
            for path in vehicle_paths.values():
                if path:
                    for point in path:
                        px, py = point
                        vis_grid[py, px] = [100, 255, 100]  # 路径绿色

        # 绘制目标点
        target_x, target_y = self.target_point
        vis_grid[target_x][target_y] = [255, 0, 0]

        # 绘制观测点
        for x, y in self.observation_point_dict.keys():
            vis_grid[x][y] = [255, 255, 0]

        # 绘制车辆位置
        for car_id in self.vehicles_dict.keys():
            x, y = self.vehicles_dict[car_id].map_location
            vis_grid[x][y] = [0, 0, 255]


        # 显示图像
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_grid, origin='lower', extent=[0, self.width, 0, self.height])
        plt.title('CityGuard')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    simulator = GridSimulator()
    simulator.visualize()
    key_lst = list(simulator.vehicles_dict.keys())
    path = simulator.vehicle_id_find_path(key_lst[0], simulator.target_point)
    print(path)
    simulator.visualize({key_lst[0]: path})