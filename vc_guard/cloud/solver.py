import heapq
import json
import random
import uuid

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from env_utils.llm_args import *
from vc_guard.common.constants import VEHICLE_SIMPLE_REPORT_KEY
from vc_guard.common.models import ParseUserPromptVo, SummaryVo, BestVehicleIdListVo
from vc_guard.common.prompts import parse_user_prompt_template, multi_view_understanding_summary_template, \
    get_best_vehicle_id_list_template
from vc_guard.edge.vechicle import AgentCard
from vc_guard.globals.executor import vehicle_executor
from vc_guard.globals.memory import long_term_memory_store
from vc_guard.globals.vehicles import vehicles
from vc_guard.grid.map import MapSimulator, latlon_to_grid


class CloudSolver:
    """
    云端计算
    """
    def __init__(self):
        self.llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model)
        self.map_simulator = MapSimulator(width=30, height=30)  # 创建地图模拟器

    def _parse_user_prompt(self, user_prompt: str, is_log: bool) -> ParseUserPromptVo:
        """
        解析用户输入，从用户输入中提取出需要执行的任务以及对应的位置坐标信息
        :param user_prompt: 用户输入
        :param is_log: 是否打印日志
        :return: 任务 + 位置对象
        """

        def get_location_xy(location: str) -> tuple[float, float]:
            """
            解析位置坐标信息，返回经纬度坐标
            :param location: 位置信息
            :return: 经纬度坐标元组
            """
            # TODO : 这里需要根据实际情况调用第三方 API 或者模拟器 API 解析位置信息，这里简单模拟
            return random.uniform(-90, 90), random.uniform(-180, 180)

        agent = create_agent(
            model=self.llm,
            tools=[get_location_xy],
            response_format=ToolStrategy(ParseUserPromptVo)
        )
        prompt = parse_user_prompt_template.format(user_input=user_prompt)
        response = agent.invoke({"messages": [prompt]})

        if is_log:
            print(f"parse_user_prompt ===> ")
            print(f"prompt: {prompt}")
            print(f"response: {response["structured_response"]}")
            print(f"parse_user_prompt <===")
            print()

        return response["structured_response"]

    def _get_agent_cards(self, location: tuple[float, float], is_log: bool) -> list[str]:
        """
        获取任务执行地点附近所有车辆的 agent_card
        :param location: 任务执行地点
        :param is_log: 是否打印日志
        :return: agent_card 列表
        """
        # TODO 这里的 vehicle_list 实际上需要从模拟器中利用 location 获取附近的车辆，这里直接模拟
        car_id_set: set[str] = set([vehicle.car_id for vehicle in vehicles])

        # 线程池并行处理
        agent_cards: list[str] = vehicle_executor.execute_tasks(
            vehicle_list=vehicles,
            best_vehicle_id_set=car_id_set,
            method_name='get_agent_card',
            return_results=True
        )

        if is_log:
            print(f"get_agent_cards ===> ")
            # print(f"location: {location}")
            print(f"agent_cards: {agent_cards}")
            print(f"get_agent_cards <===")
            print()

        return agent_cards

    def _init_map_simulator(self,
                            location: tuple[float, float],
                            agent_cards: list[str],
                            is_log: bool) -> list[AgentCard]:
        """
        初始化地图模拟器agent_card_models: list[AgentCard],
        :param location: 目标位置
        :param agent_cards: 车辆信息
        :param is_log: 是否打印日志
        :return: 车辆对象信息
        """

        # 将任务地点坐标映射到模拟的网格地图上
        task_location = latlon_to_grid(
            location[0], location[1],
            self.map_simulator.width, self.map_simulator.height
        )

        self.map_simulator.add_target_point(task_location)

        agent_card_models = [AgentCard(**json.loads(agent_card)) for agent_card in agent_cards]

        # 将车辆坐标映射到模拟的网格地图上
        for agent_card_model in agent_card_models:
            # 坐标映射
            agent_card_model.location = latlon_to_grid(
                agent_card_model.location[0], agent_card_model.location[1],
                self.map_simulator.width, self.map_simulator.height
            )

        # 网格图中添加车辆
        self.map_simulator.add_vehicle([agent_card_model.location for agent_card_model in agent_card_models])

        if is_log:
            self.map_simulator.visualize()

        return agent_card_models

    def _get_best_vehicle_id_list(self,
                                  agent_card_models: list[AgentCard],
                                  num_of_vehicles: int,
                                  is_log: bool) -> list[str]:
        """
        挑选最优的车辆执行任务
        :param agent_card_models: agent_card 对象列表
        :param num_of_vehicles: 执行任务的车辆数量
        :return: 车辆 id 列表
        """
        # 使用大模型筛选车辆
        agent = create_agent(model=self.llm, tools=[], response_format=ToolStrategy(BestVehicleIdListVo))

        prompt = get_best_vehicle_id_list_template.format(
            grid_matrix=self.map_simulator.grid_matrix,
            task_location=self.map_simulator.target_point,
            num_of_vehicles=num_of_vehicles,
            agent_card_models=agent_card_models
        )

        response = agent.invoke({"messages": [prompt]})

        # 拿到最优车辆的 id 列表
        best_vehicle_id_list_vo = response["structured_response"]

        best_vehicle_id_list = best_vehicle_id_list_vo.best_vehicle_id_list

        if is_log:
            print(f"get_best_vehicle_id_list ===> ")
            print(f"best_vehicle_id_list: {best_vehicle_id_list}")
            print(f"get_best_vehicle_id_list <===")
            print()

        return best_vehicle_id_list

    def _plan_path(self,
                   agent_card_models: list[AgentCard],
                   best_vehicle_id_set: set[str],
                   is_log: bool):
        # 拿到坐标信息
        vehicle_locations = [agent_card_model.location
                             for agent_card_model in agent_card_models
                             if agent_card_model.car_id in best_vehicle_id_set]

        vehicle_paths = {}

        for i, position in enumerate(vehicle_locations):
            vehicle_id = f"Vehicle-{i + 1}"
            path = self.map_simulator.find_path(position, self.map_simulator.target_point)
            vehicle_paths[vehicle_id] = path

        if is_log:
            self.map_simulator.visualize(vehicle_paths=vehicle_paths)


    def query(self,
              user_prompt: str,
              num_of_vehicles: int = 3,
              is_log: bool = False) -> SummaryVo:
        """
        云端下发查询，返回结果报告
        :param num_of_vehicles: 执行任务的车辆数量
        :param user_prompt: 用户输入
        :param is_log: 是否打印日志
        :return: 结果报告
        """
        # 生成本次任务的 uuid，用于 Memory 存储
        task_uuid = "task-" + str(uuid.uuid4()).replace("-", "")

        # 1. 云端 LLM 解析用户输入
        parse_user_prompt_vo = self._parse_user_prompt(user_prompt, is_log)

        # 2. 云端下发广播，所有车辆返回自身的 agent_card
        agent_cards = self._get_agent_cards(parse_user_prompt_vo.location, is_log)


        agent_card_models = self._init_map_simulator(parse_user_prompt_vo.location, agent_cards, is_log)

        # 3. 挑选最优的车辆执行任务
        best_vehicle_id_list = self._get_best_vehicle_id_list(agent_card_models, num_of_vehicles, is_log)
        if not best_vehicle_id_list:
            raise Exception("没有找到合适的车辆执行任务")

        best_vehicle_id_set = set(best_vehicle_id_list)
        task_description = parse_user_prompt_vo.task

        # 路径规划
        self._plan_path(agent_card_models, best_vehicle_id_set, is_log)

        # 4. 执行任务，线程池并行模拟
        vehicle_executor.execute_tasks(
            vehicle_list=vehicles,
            best_vehicle_id_set=best_vehicle_id_set,
            method_name='execute_task',
            args=(parse_user_prompt_vo.location, task_description, task_uuid, is_log)
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

        if is_log:
            print(f"summary ===> ")
            print(f"prompt: {prompt}")
            print(f"summary_list: {simple_report_list}")
            print(f"summary <===")
            print()

        final_report = response["structured_response"]

        return final_report