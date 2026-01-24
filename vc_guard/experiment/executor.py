import csv
import os
import random
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from langchain.agents.structured_output import ToolStrategy

from env_utils.llm_args import *
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from vc_guard.common.constants import VEHICLE_SIMPLE_REPORT_KEY
from vc_guard.common.models import HandleObservationVo, ParseUserPromptVo, SummaryVo
from vc_guard.common.prompts import parse_user_prompt_template, multi_view_understanding_summary_template
from vc_guard.globals.executor import vehicle_executor
from vc_guard.globals.grid import map_simulator
from vc_guard.globals.memory import long_term_memory_store
from vc_guard.grid.map import MapSimulator, get_quadrant
from vc_guard.observations.handlers import get_project_root, BaseObservationHandler, SimpleImageObservationHandler, \
    ImageObservationHandler, MapImageObservationHandler


class ExperimentVehicle:
    """
    实验车辆，仅保留实验用所需的属性
    """
    def __init__(self, map_width: int = 30, map_height: int = 30):
        # 车辆属性赋值
        self.car_id: str = uuid.uuid4().__str__()
        self.direction: float = random.uniform(0, 360)
        self.observation_handler = ImageObservationHandler()  # 车辆观测处理器
        self.location: tuple[int, int] = (random.randint(0, map_height), random.randint(0, map_width))
        self.speed: float = random.uniform(30, 60)
        self.direction: float = random.uniform(0, 360)
        # 车辆内置 Agent
        self.agent = create_agent(
            model=ChatOpenAI(api_key=api_key, base_url=base_url, model=visual_model),
            tools=[],
            response_format=ToolStrategy(HandleObservationVo)
        )

    def set_observation_handler(self, observation_handler: BaseObservationHandler):
        self.observation_handler = observation_handler

    def execute_map_task(self, map_simulator, task_location, task_description, task_uuid, type_name, file_id, is_log):
        observation, timestamp = self.observation_handler.get_observation(map_simulator, self.location, type_name, file_id)
        simple_report = self.observation_handler.handle_observation(
            agent=self.agent,
            observation=observation,
            timestamp=timestamp,
            task_location=task_location,
            task_description=task_description,
            task_uuid=task_uuid,
            car_id=self.car_id,
            car_direction=self.direction
        )

        # 车辆生成完本次的报告后，保存报告到云上
        long_term_memory_store.rpush(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid), simple_report)

        if is_log:
            print(f"execute_map_task ===> ")
            print(f"car_id: {self.car_id}")
            print(f"simple_report: {simple_report}")
            print(f"execute_map_task <===")
            print()

    def execute_task(self,
                     task_location: str,
                     task_description: str,
                     task_uuid: str,
                     dataset_path,
                     type_name: str,
                     file_id: str,
                     is_log: bool = False
                     ) -> None:
        # 调用观测处理器处理观测
        observation, timestamp = self.observation_handler.get_observation(dataset_path, type_name, file_id)
        simple_report = self.observation_handler.handle_observation(
            agent=self.agent,
            observation=observation,
            timestamp=timestamp,
            task_location=task_location,
            task_description=task_description,
            task_uuid=task_uuid,
            car_id=self.car_id,
            car_direction=self.direction
        )

        # 车辆生成完本次的报告后，保存报告到云上
        long_term_memory_store.rpush(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid), simple_report)

        if is_log:
            print(f"execute_task ===> ")
            print(f"car_id: {self.car_id}")
            # print(f"observation: {observation}")
            print(f"simple_report: {simple_report}")
            # print(f"long_term_list_memory: {long_term_memory.list_memory}")
            print(f"execute_task <===")
            print()

class BatchExperimentExecutor(ABC):
    """
    批量实验执行器
    """
    def __init__(self,
                 user_prompt: str,
                 output_csv: str,
                 type_name: str = "garbage",
                 num_vehicles: int = 3):
        self.user_prompt = user_prompt
        self.output_csv = Path(output_csv)    # 输出 csv 文件路径
        self.type_name = type_name
        self.num_vehicles = num_vehicles
        self.llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model)

    def _get_csv_headers(self) -> list[str]:
        """
        获取 csv 文件表头
        :return: 获取 csv 文件表头
        """
        return ["type_name", "file_id", "summary"]

    def _int_csv(self) -> None:
        """
        初始化 csv 文件并写入表头
        :return:
        """
        if not self.output_csv.exists():
            with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._get_csv_headers())
                writer.writeheader()

    def _save_to_csv(self, results: list[dict[str, str]]):
        """
        将结果保存到 csv 文件中
        :param results: 结果列表
        :return: None
        """
        with open(self.output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_csv_headers())
            writer.writerows(results)

    def _count_subdirectories(self, folder_path: str) -> int:
        """
        计算指定文件夹下的子文件夹数量
        :param folder_path: 目标文件夹路径
        :return: 子文件夹数量 (如果路径不存在或不是文件夹返回 0)
        """

        folder_path = os.path.join(folder_path, self.type_name)
        # 转换为Path对象
        path = Path(folder_path)

        # 检查路径是否存在且是文件夹
        if not path.exists() or not path.is_dir():
            return 0

        # 获取所有子项并筛选出文件夹
        subdirectories = [item for item in path.iterdir() if item.is_dir()]

        return len(subdirectories)

    def _parse_user_prompt(self) -> ParseUserPromptVo:
        """
        解析用户输入，从用户输入中提取出需要执行的任务以及对应的位置坐标信息
        :return: 任务 + 位置对象
        """
        agent = create_agent(
            model=self.llm,
            tools=[],
            response_format=ToolStrategy(ParseUserPromptVo)
        )
        prompt = parse_user_prompt_template.format(user_input=self.user_prompt)
        response = agent.invoke({"messages": [prompt]})

        return response["structured_response"]

    def _multi_view_understand(self,
                               simple_report_list: list,
                               task_description: str,
                               task_uuid: str,
                               is_log: bool,
                               ) -> SummaryVo:
        prompt = multi_view_understanding_summary_template.format(
            simple_report_list=simple_report_list,
            task_id=task_uuid,
            task_description=task_description
        )

        agent = create_agent(model=self.llm, tools=[], response_format=ToolStrategy(SummaryVo))

        # 6. 解析输出
        response = agent.invoke({"messages": [prompt]})

        if is_log:
            print(f"summary ===> ")
            print(f"prompt: {prompt}")
            print(f"summary_list: {simple_report_list}")
            print(f"summary <===")
            print()

        final_report = response["structured_response"]
        return final_report

    @abstractmethod
    def run_experiment(self, start_idx: int) -> None:
        """
        运行实验
        :param start_idx: 起始文件夹编号
        :return:
        """
        pass

class SingleAndMultiViewExperiment(BatchExperimentExecutor):
    """
    单视角和多视角实验
    """
    def _run_single_experiment(self,
                               task_uuid: str,
                               experiment_vehicle_id_set: set[str],
                               experiment_vehicles: list[ExperimentVehicle],
                               dataset_path,
                               file_id: int,
                               task_description: str,
                               task_location: str) -> str:
        # 车辆先执行任务
        vehicle_executor.execute_tasks(
            vehicle_list=experiment_vehicles,
            best_vehicle_id_set=experiment_vehicle_id_set,
            method_name='execute_task',
            args=(task_location, task_description, task_uuid, dataset_path, self.type_name, str(file_id))
        )

        # 获取车辆的简单报告列表
        simple_report_list = long_term_memory_store.get_list(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid))

        # 5. 多视角理解
        prompt = multi_view_understanding_summary_template.format(
            simple_report_list=simple_report_list,
            task_id=task_uuid,
            task_description=task_description
        )

        agent = create_agent(model=self.llm, tools=[], response_format=ToolStrategy(SummaryVo))

        # 6. 解析输出
        response = agent.invoke({"messages": [prompt]})

        final_report = response["structured_response"]

        return final_report.summary

    def run_experiment(self, start_idx: int) -> None:

        # 创建车辆列表
        experiment_vehicles = [ExperimentVehicle() for _ in range(self.num_vehicles)]
        experiment_vehicle_id_set = set([vehicle.car_id for vehicle in experiment_vehicles])

        # 解析用户输入
        parse_user_prompt_vo = self._parse_user_prompt()
        task_location = parse_user_prompt_vo.task_location
        task_description = parse_user_prompt_vo.task

        # 获取数据集根目录
        project_root = get_project_root()

        # 拼接目标目录路径
        dataset_path = os.path.join(project_root, "datasets")

        # 获取整批次的数量
        total_cnt = self._count_subdirectories(dataset_path)
        # print(total_cnt)

        # 初始化 csv 文件
        self._int_csv()

        # TODO 先串行执行所有任务
        for i in range(start_idx, total_cnt + 1):
            task_uuid = "task-" + str(uuid.uuid4()).replace("-", "")

            summary = self._run_single_experiment(
                task_uuid,
                experiment_vehicle_id_set,
                experiment_vehicles,
                dataset_path,
                i,
                task_description,
                task_location
            )

            result = {
                "type_name": self.type_name,
                "file_id": str(i),
                "summary": summary
            }

            self._save_to_csv([result])

class ActiveSchedulingExperiment(BatchExperimentExecutor):
    """
    主动调度实验
    """
    def _get_csv_headers(self) -> list[str]:
        """
        获取 csv 文件表头
        :return: 获取 csv 文件表头
        """
        return ["type_name", "file_id", "summary1", "summary2"]

    def _run_single_experiment(self,
                               task_uuid: str,
                               map_simulator: MapSimulator,
                               nearby_radius: int,
                               experiment_vehicles_dict: dict,
                               type_name: str,
                               file_id: int,
                               task_description: str,
                               task_location: str) -> str:
        # 拿到附近车辆的 id - 位置字典
        nearby_vehicle_id_position_dict: dict[str, tuple[int, int]] = map_simulator.get_nearby_vehicle_id_position_dict(nearby_radius)

        # 整理象限集合
        quadrants = set()
        for car_id, position in nearby_vehicle_id_position_dict.items():
            quadrants.add(
                get_quadrant(position[0], position[1], map_simulator.target_point[0], map_simulator.target_point[1]))

        final_report = None

        # 附近车辆报告
        vehicle_executor.execute_tasks(
            vehicle_list=experiment_vehicles_dict.values(),
            best_vehicle_id_set=set(nearby_vehicle_id_position_dict.keys()),
            method_name='execute_map_task',
            args=(map_simulator, task_location, task_description, task_uuid, type_name, str(file_id), False)
        )

        simple_report_list = long_term_memory_store.get_list(VEHICLE_SIMPLE_REPORT_KEY.format(task_uuid=task_uuid))

        summaryVo = self._multi_view_understand(
            simple_report_list=simple_report_list,
            task_description=task_description,
            task_uuid=task_uuid,
            is_log=True
        )

        return summaryVo.summary

    def run_experiment(self, start_idx: int) -> None:
        # 解析用户输入
        parse_user_prompt_vo = self._parse_user_prompt()
        task_location = parse_user_prompt_vo.task_location
        task_description = parse_user_prompt_vo.task

        # 获取数据集根目录
        project_root = get_project_root()

        # 拼接目标目录路径
        dataset_path = os.path.join(project_root, "datasets")

        # 获取整批次的数量
        total_cnt = self._count_subdirectories(dataset_path)

        # 初始化 csv 文件
        self._int_csv()

        # TODO 先串行执行所有任务
        for i in range(start_idx, total_cnt + 1):
            # 创建地图模拟器
            map_simulator = MapSimulator(width=30, height=30)

            # 创建车辆列表，拼接车辆字典
            observation_handler = MapImageObservationHandler()
            experiment_vehicles = [ExperimentVehicle() for _ in range(20)]
            experiment_vehicles_dict = {}
            for vehicle in experiment_vehicles:
                vehicle.observation_handler = observation_handler
                experiment_vehicles_dict[vehicle.car_id] = vehicle

            # 添加目标点
            map_simulator.add_vehicle_id_position(experiment_vehicles_dict)
            map_simulator.add_target_point((15, 15))

            map_simulator.visualize()

            task_uuid = "task-" + str(uuid.uuid4()).replace("-", "")

            summary = self._run_single_experiment(
                task_uuid,
                map_simulator,
                5,
                experiment_vehicles_dict,
                self.type_name,
                i,
                task_description,
                task_location
            )

            result = {
                "type_name": self.type_name,
                "file_id": str(i),
                "summary1": summary,
                "summary2": "null"
            }

            self._save_to_csv([result])

if __name__ == "__main__":
    # 经市民举报，长沙市岳麓区阜埠河路附近存在较大异味，请查询根因。
    # 经市民举报，长沙市岳麓区阜埠河路附近机动车道拥挤，有时伴有行人通过，请查询根因。
    # 经市民举报，长沙市岳麓区阜埠河路附近消防车拥堵，无法及时赶到火灾现场，请分析根因。
    # 经市民举报，长沙市岳麓区阜埠河路附近出现大量烟雾，味道刺鼻，且阻挡驾驶视野，请分析根因。
    # 经市民举报，长沙市岳麓区阜埠河路附近道路出现拥挤，行人和机动车相互穿插，道路拥堵。
    # 经市民举报，长沙市岳麓区阜埠河路附近道路出现积水情况，请分析根因。
    user_prompt = "经市民举报，长沙市岳麓区阜埠河路附近存在较大异味，请查询根因。"
    output_csv = "unactive_garbage_output.csv"
    type_name = "garbage"
    num_vehicles = 2

    # executor = SingleAndMultiViewExperiment(user_prompt, output_csv, type_name, num_vehicles)
    executor = ActiveSchedulingExperiment(user_prompt, output_csv, type_name, num_vehicles)
    executor.run_experiment(start_idx=1)